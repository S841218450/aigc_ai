"""
知识库文档处理编排服务
========================
职责（service 层，供薄 HTTP 层调用）：
- process_uploaded_file：统一文件处理编排（Excel 行级块化 / DOCX 双流水线 / PDF·TXT·MD 语义分块 / 纯文本兜底）
- ingest_document：文档入库（参数校验 → 幂等跳过 → 元数据登记 → 分割 → 向量入库 → 状态回写）
- delete_document / batch_delete_documents：物理/软删除（向量库 + 元数据强一致）
- list_documents：文档列表（排障/对账）

调用方：app/api/v1/endpoints/knowledge_base.py
依赖：
- 元数据 CRUD + Java 回调：app/services/knowledge_base_service.py
- 文件下载/解析工具：app/tools/common/file_download.py
- 向量库 / 文档处理工具：app/tools/retrieval/
"""
import asyncio
import traceback
from typing import List, Dict

from app.models.schemas.knowledge_base import (
    BatchDeleteItemResult,
    BatchDeleteRequest,
    BatchDeleteResponse,
    DeleteResponse,
    DocumentListResponse,
    KnowledgeBaseUploadRequest,
    KnowledgeBaseUploadResponse,
)
from app.services.knowledge_base_service import (
    KB_METADATA_SERVICE,
    change_knowledge_doc_status,
)
from app.tools.common.file_download import (
    DOCX_SUFFIXES,
    EXCEL_SUFFIXES,
    cleanup_tmp,
    detect_file_type,
    download_and_extract_content,
    download_file_to_tmp,
)
from app.tools.common.table_registry import TABLE_REGISTRY
from app.tools.retrieval.document_processor import DocumentProcessor
from app.tools.retrieval.vector_store import VectorStoreTool
from app.utils.logger_handle import logger


# ---------------------------------------------------------------------------
# 1. 统一文件处理编排（入库专用）
# ---------------------------------------------------------------------------

async def process_uploaded_file(
    file_url: str,
    *,
    doc_id: str,
    doc_name: str,
    extra_metadata: dict = None,
    file_name_hint: str = "",
    file_type_hint: str = "",
    fallback_content: str = "",
):
    """
    统一文件处理（file_url 优先，content 兜底）：
    - Excel (.xlsx/.xls)：下载到临时文件 → 行级块化（process_excel_file，不走语义分块）
    - DOCX：下载到临时文件 → 段落流 + 表格流双流水线（process_docx_file）
    - PDF/TXT/MD：下载 → 提取纯文本 → 语义分块（process_document）
    - 兜底：纯文本 content → 语义分块

    Args:
        file_url: 对象存储 URL（可为空）
        doc_id/doc_name/extra_metadata: 透传给 DocumentProcessor 做 chunk 元数据
        file_name_hint: doc_name，用于兜底推断类型
        file_type_hint: Java 端传的文件类型（可选）
        fallback_content: 纯文本兜底内容（file_url 为空时使用）

    Returns:
        (processed: dict, file_type: str, file_size: int, file_md5: str)

    Raises:
        ValueError: 下载/解析失败、类型不支持、内容为空
    """
    processor = DocumentProcessor()

    if file_url:
        # 先按 URL 后缀推断类型；URL 无后缀时用 doc_name 兜底
        suffix, _ = detect_file_type(file_url, default_type=file_type_hint)
        if not suffix and file_name_hint:
            suffix, _ = detect_file_type(file_name_hint, default_type=file_type_hint)

        # Excel：下载到临时文件 → 行级块化
        if suffix in EXCEL_SUFFIXES:
            tmp_dir, tmp_path, _suffix, file_type, file_size, file_md5 = await download_file_to_tmp(
                file_url, file_name_hint=file_name_hint, file_type_hint=file_type_hint,
            )
            try:
                processed = await asyncio.to_thread(
                    processor.process_excel_file, tmp_path, doc_name, doc_id, extra_metadata,
                )
            finally:
                cleanup_tmp(tmp_dir, tmp_path)
            return processed, file_type, file_size, file_md5

        # DOCX：下载到临时文件 → 段落+表格双流水线
        if suffix in DOCX_SUFFIXES:
            tmp_dir, tmp_path, _suffix, file_type, file_size, file_md5 = await download_file_to_tmp(
                file_url, file_name_hint=file_name_hint, file_type_hint=file_type_hint,
            )
            try:
                processed = await asyncio.to_thread(
                    processor.process_docx_file, tmp_path, doc_name, doc_id, extra_metadata,
                )
            finally:
                cleanup_tmp(tmp_dir, tmp_path)
            return processed, file_type, file_size, file_md5

        # PDF / TXT / MD：下载 → 提取纯文本 → 语义分块
        content, file_type, file_size, file_md5 = await download_and_extract_content(
            file_url, file_name_hint=file_name_hint, file_type_hint=file_type_hint,
        )
        processed = await asyncio.to_thread(
            processor.process_document, content, doc_name, doc_id, extra_metadata,
        )
        return processed, file_type, file_size, file_md5

    # 兜底：纯文本 content
    if not fallback_content:
        raise ValueError("file_url 和 content 均为空，无法入库")
    processed = await asyncio.to_thread(
        processor.process_document, fallback_content, doc_name, doc_id, extra_metadata,
    )
    return processed, (file_type_hint or "txt"), 0, ""


# ---------------------------------------------------------------------------
# 2. 文档入库
# ---------------------------------------------------------------------------

# 后台入库任务注册表：doc_id → asyncio.Task
# 作用：1) 防止同一 doc_id 重复提交启动多个并发任务；2) 持有引用防止 task 被 GC 提前回收
_running_ingest_tasks: Dict[str, asyncio.Task] = {}


async def _run_ingest_task(doc_id: str, payload: KnowledgeBaseUploadRequest) -> None:
    """后台执行入库主体；异常兜底回调 failed；结束移除任务注册。"""
    try:
        await ingest_document(payload)
    except Exception as e:
        # ingest_document 内部已全量捕获业务异常，此处仅兜底极端情况
        logger.error(f"[KB入库] 后台任务异常 doc_id={doc_id} err={e}\n{traceback.format_exc()}")
        try:
            await change_knowledge_doc_status(
                doc_id, "failed", fail_reason=f"后台任务异常: {str(e)[:300]}",
            )
        except Exception as ex:
            logger.warning(f"[KB入库] 后台任务失败回调异常 doc_id={doc_id} err={ex}")
    finally:
        _running_ingest_tasks.pop(doc_id, None)


async def submit_ingest_document(payload: KnowledgeBaseUploadRequest) -> KnowledgeBaseUploadResponse:
    """
    ### 提交即返回 200（解决 Java 端同步调用大文件/Excel 入库超时误判失败）

    收到请求立即响应，后台异步执行入库，完成后由 ingest_document 内部回调
    PUT /api/ai/knowledge/status 更新 Java 端状态（processing → ready/failed）。

    流程：
    1. 快速校验：user_id 空 / file_url、content 均空 → 立即返回 failed（不启动后台任务）
    2. 幂等快速路径：doc_id 已 ready 且非 force_reingest → 立即返回 skipped
    3. 去重：同一 doc_id 已有运行中任务 → 返回"已提交处理中"，不重复启动
    4. 其余：asyncio.create_task 后台执行 ingest_document，接口立即返回 status=processing
    """
    # 1) 快速校验
    if not payload.user_id:
        resp = KnowledgeBaseUploadResponse(
            success=False,
            doc_id=payload.doc_id,
            status="failed",
            message="user_id 不能为空，上传文档必须指定归属用户",
        )
        await change_knowledge_doc_status(
            payload.doc_id, "failed",
            fail_reason=resp.message,
            file_size=int(payload.file_size or 0),
        )
        return resp

    if not payload.file_url and not payload.content:
        resp = KnowledgeBaseUploadResponse(
            success=False,
            doc_id=payload.doc_id,
            status="failed",
            message="file_url 和 content 不能同时为空，请至少提供一种内容来源",
        )
        await change_knowledge_doc_status(
            payload.doc_id, "failed",
            fail_reason=resp.message,
            file_size=int(payload.file_size or 0),
        )
        return resp

    # 2) 幂等快速路径（已 ready 且非强制重传 → 跳过，不启动后台任务）
    existing = await KB_METADATA_SERVICE.get_document(payload.doc_id)
    if existing and existing.get("status") == "ready" and not payload.force_reingest:
        sk_chunks = int(existing.get("chunk_count", 0) or 0)
        resp = KnowledgeBaseUploadResponse(
            success=True,
            skipped=True,
            skip_reason="doc_id 已入库且状态 ready，未启用 force_reingest",
            doc_id=payload.doc_id,
            chunk_count=sk_chunks,
            file_type=existing.get("file_type", ""),
            file_md5=existing.get("file_md5", ""),
            file_size=int(existing.get("file_size", 0) or 0),
            status="skipped",
            message="已跳过（未变更）。如需重新处理，请传 force_reingest=true。",
        )
        await change_knowledge_doc_status(
            payload.doc_id, "ready",
            chunk_count=sk_chunks,
            file_size=resp.file_size,
        )
        logger.info(f"[KB入库] 幂等跳过 doc_id={payload.doc_id} chunks={sk_chunks}")
        return resp

    # 3) 去重：同一 doc_id 已有运行中任务
    running = _running_ingest_tasks.get(payload.doc_id)
    if running and not running.done():
        return KnowledgeBaseUploadResponse(
            success=True,
            doc_id=payload.doc_id,
            status="processing",
            message="已提交，该 doc_id 正在后台处理中（重复提交，等待状态回调即可）",
        )

    # 4) 启动后台任务，立即返回
    task = asyncio.create_task(_run_ingest_task(payload.doc_id, payload))
    _running_ingest_tasks[payload.doc_id] = task
    logger.info(f"[KB入库] 已提交后台处理 doc_id={payload.doc_id} doc_name={payload.doc_name}")
    return KnowledgeBaseUploadResponse(
        success=True,
        doc_id=payload.doc_id,
        status="processing",
        file_type=payload.file_type or "",
        file_md5=payload.file_md5 or "",
        file_size=int(payload.file_size or 0),
        message="已提交，正在后台处理中",
    )


async def ingest_document(payload: KnowledgeBaseUploadRequest) -> KnowledgeBaseUploadResponse:
    """
    ### 调用方：Java 业务端（用户在 Java 端完成文件上传到 COS/OSS 后回调此接口）

    ### 流程：
    1. 参数校验（file_url 和 content 至少一个存在；user_id 必传）
    2. 幂等检查：同一 doc_id 已 ready 且非强制 → 直接返回，省 embedding 成本
    3. 写 MongoDB 元数据（status=processing）
    4. 统一文件处理（Excel 旁路行级块化，PDF/TXT/MD 走下载→提取→语义分块，兜底纯文本）
    5. VectorStoreTool.upsert_chunks → 写入 Chroma（同 doc_id 先删旧 chunk 再插入）
    6. 成功：mark_ready；失败：mark_failed 并保留错误原因；同步回写 Java 状态
    """
    # 1) 参数校验
    if not payload.user_id:
        # 用户隔离维度缺失：拒绝入库，避免产生无主数据（后续检索强过滤后会被漏掉）
        resp = KnowledgeBaseUploadResponse(
            success=False,
            doc_id=payload.doc_id,
            status="failed",
            message="user_id 不能为空，上传文档必须指定归属用户",
        )
        await change_knowledge_doc_status(
            payload.doc_id, "failed",
            fail_reason=resp.message,
            file_size=int(payload.file_size or 0),
        )
        return resp

    if not payload.file_url and not payload.content:
        resp = KnowledgeBaseUploadResponse(
            success=False,
            doc_id=payload.doc_id,
            status="failed",
            message="file_url 和 content 不能同时为空，请至少提供一种内容来源",
        )
        await change_knowledge_doc_status(
            payload.doc_id, "failed",
            fail_reason=resp.message,
            file_size=int(payload.file_size or 0),
        )
        return resp

    # 2) 幂等跳过
    existing = await KB_METADATA_SERVICE.get_document(payload.doc_id)
    if existing and existing.get("status") == "ready" and not payload.force_reingest:
        sk_chunks = int(existing.get("chunk_count", 0) or 0)
        resp = KnowledgeBaseUploadResponse(
            success=True,
            skipped=True,
            skip_reason="doc_id 已入库且状态 ready，未启用 force_reingest",
            doc_id=payload.doc_id,
            chunk_count=sk_chunks,
            file_type=existing.get("file_type", ""),
            file_md5=existing.get("file_md5", ""),
            file_size=int(existing.get("file_size", 0) or 0),
            status="skipped",
            message="已跳过（未变更）。如需重新处理，请传 force_reingest=true。",
        )
        await change_knowledge_doc_status(
            payload.doc_id, "ready",
            chunk_count=sk_chunks,
            file_size=resp.file_size,
        )
        logger.info(f"[KB入库] 幂等跳过 doc_id={payload.doc_id} chunks={sk_chunks}")
        return resp

    # 修改状态为开始处理
    await change_knowledge_doc_status(payload.doc_id, "processing")

    # 3) 先登记元数据（processing），避免重复请求并发入库
    ok = await KB_METADATA_SERVICE.create_document(
        doc_id=payload.doc_id,
        doc_name=payload.doc_name,
        file_url=payload.file_url or "",
        file_md5=payload.file_md5 or "",
        file_size=int(payload.file_size or 0),
        file_type=payload.file_type or "",
        metadata=payload.metadata or {},
        kb_id=payload.kb_id or "default",
    )
    if not ok:
        resp = KnowledgeBaseUploadResponse(
            success=False,
            doc_id=payload.doc_id,
            status="failed",
            message="元数据登记失败（MongoDB 写入异常）",
        )
        await change_knowledge_doc_status(
            payload.doc_id, "failed",
            fail_reason=resp.message,
            file_size=int(payload.file_size or 0),
        )
        return resp

    resolved_file_type = payload.file_type or ""
    resolved_file_size = int(payload.file_size or 0)
    resolved_file_md5 = payload.file_md5 or ""

    try:
        # 4) 统一文件处理（Excel 旁路行级块化，PDF/TXT/MD 走 download→提取→语义分块，兜底纯文本）
        try:
            processed, resolved_file_type, resolved_file_size, resolved_file_md5 = (
                await process_uploaded_file(
                    payload.file_url or "",
                    doc_id=payload.doc_id,
                    doc_name=payload.doc_name,
                    # metadata 只传 Java 真正有的：folder_id 等业务元数据
                    # file_url / ingest_source 是展示/调试用，不进 chunk metadata（对检索 0 帮助）
                    extra_metadata={
                        **(payload.metadata or {}),
                        "kb_id": payload.kb_id or "default",
                        "owner_id": payload.user_id,
                    },
                    file_name_hint=payload.doc_name,
                    file_type_hint=payload.file_type or "",
                    fallback_content=payload.content or "",
                )
            )
            chunks = processed.get("chunks", [])
        except ValueError as e:
            fail_reason = f"下载/解析失败: {e}"
            logger.warning(f"[KB入库] 下载/解析失败 doc_id={payload.doc_id} reason={fail_reason}")
            await KB_METADATA_SERVICE.mark_failed(payload.doc_id, fail_reason)
            resp = KnowledgeBaseUploadResponse(
                success=False,
                doc_id=payload.doc_id,
                status="failed",
                file_type=resolved_file_type,
                file_md5=resolved_file_md5,
                file_size=resolved_file_size,
                message=str(e),
            )
            await change_knowledge_doc_status(
                payload.doc_id, "failed",
                fail_reason=fail_reason,
                file_size=resp.file_size,
            )
            return resp

        if not chunks:
            raise ValueError("文档分割后无任何有效 chunk，可能文件内容全为空或被过滤")

        # 6) 向量入库（同一 doc_id 先删旧的再插入，幂等）
        vector = VectorStoreTool()
        upsert_result = await vector.upsert_chunks(chunks, replace_existing_doc=True)
        inserted = int(upsert_result.get("inserted_count", 0) or 0)

        # 6.1) 结构化表登记（Excel/docx 表格 → 表目录 + 行数据，供枚举/推荐类问答确定性查询）
        # 结构化是增强能力，登记失败不阻断向量入库
        if processed.get("structured_tables"):
            try:
                await TABLE_REGISTRY.save_tables_for_doc(
                    doc_id=payload.doc_id,
                    doc_name=payload.doc_name,
                    kb_id=payload.kb_id or "default",
                    owner_id=payload.user_id,
                    folder_id=(payload.metadata or {}).get("folder_id"),
                    source_type=resolved_file_type or "excel",
                    tables=processed["structured_tables"],
                )
            except Exception as e:
                logger.warning(f"[KB入库] 结构化登记失败 doc_id={payload.doc_id} err={e}")

        # 7) 标记 ready
        chunk_total = int(processed.get("total_chunks", 0) or len(chunks))
        await KB_METADATA_SERVICE.mark_ready(
            payload.doc_id,
            chunk_count=chunk_total,
            extra_update={
                "file_type": resolved_file_type,
                "file_md5": resolved_file_md5,
                "file_size": resolved_file_size,
            },
        )

        # Excel 响应里附加 sheet 信息，便于排障
        sheets_info = processed.get("sheet_summaries") or []
        extra_msg = ""
        if sheets_info:
            extra_msg = (
                f"；共 {len(sheets_info)} 个 Sheet，行数/块数："
                + ",".join(
                    [f"{s['sheet_name']}({s['row_count']}行→{s['chunk_count']}块)" for s in sheets_info]
                )
            )

        resp = KnowledgeBaseUploadResponse(
            success=True,
            doc_id=payload.doc_id,
            chunk_count=inserted,
            file_type=resolved_file_type,
            file_md5=resolved_file_md5,
            file_size=resolved_file_size,
            status="ready",
            message=(
                f"入库成功。分割 {chunk_total} 块 / 实际入库 {inserted} 块"
                f"，批次内去重跳过 {upsert_result.get('skipped_duplicates', 0)} 块{extra_msg}。"
            ),
        )
        await change_knowledge_doc_status(
            payload.doc_id, "ready",
            chunk_count=chunk_total,
            file_size=resp.file_size,
        )
        logger.info(
            f"[KB入库] 成功 doc_id={payload.doc_id} chunk_total={chunk_total} inserted={inserted} "
            f"file_type={resolved_file_type} size={resolved_file_size}B"
        )
        return resp

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        fail_reason = f"{err}\n{traceback.format_exc()}"
        logger.error(f"[KB入库] 异常 doc_id={payload.doc_id}\n{fail_reason}")
        await KB_METADATA_SERVICE.mark_failed(payload.doc_id, fail_reason)
        resp = KnowledgeBaseUploadResponse(
            success=False,
            doc_id=payload.doc_id,
            status="failed",
            file_type=resolved_file_type,
            file_md5=resolved_file_md5,
            file_size=resolved_file_size,
            message=f"入库失败: {err}",
        )
        await change_knowledge_doc_status(
            payload.doc_id, "failed",
            fail_reason=fail_reason.splitlines()[0],
            file_size=resp.file_size,
        )
        return resp


# ---------------------------------------------------------------------------
# 3. 单文档删除
# ---------------------------------------------------------------------------

async def delete_document(doc_id: str) -> DeleteResponse:
    """
    Java 端在删除文件成功后回调此接口，物理删除向量 chunk 和元数据，避免「幽灵 chunk」。
    """
    vector = VectorStoreTool()
    vec_result = await vector.delete_by_doc_id(doc_id)
    chunk_deleted = int(vec_result.get("deleted_chunk_count", 0) or 0)

    count = await KB_METADATA_SERVICE.hard_delete(doc_id)
    deleted_meta = count > 0

    # 联动清理结构化表数据（best-effort，失败不阻断删除）
    try:
        await TABLE_REGISTRY.delete_by_doc_ids([doc_id])
    except Exception as e:
        logger.warning(f"[KB删除] 结构化表清理失败 doc_id={doc_id} err={e}")

    success = bool(vec_result.get("deleted", True))
    if not success:
        return DeleteResponse(
            success=False,
            doc_id=doc_id,
            deleted_chunk_count=chunk_deleted,
            deleted_metadata=deleted_meta,
            message=vec_result.get("reason", "向量库删除失败"),
        )

    return DeleteResponse(
        success=True,
        doc_id=doc_id,
        deleted_chunk_count=chunk_deleted,
        deleted_metadata=deleted_meta,
        message=(
            f"删除成功：清理向量 chunk {chunk_deleted} 条；"
            f"元数据已物理删除（affected={count}）"
        ),
    )


# ---------------------------------------------------------------------------
# 4. 批量删除
# ---------------------------------------------------------------------------

async def batch_delete_documents(payload: BatchDeleteRequest) -> BatchDeleteResponse:
    """
    与单删等价，但批量时只发一次 HTTP 请求，省 N 次往返。
    - 先一次性删向量（Chroma where $in，单次请求）
    - 再按 mode 批量处理元数据
    - 最后逐条回填 details，便于 Java 端对账/日志
    """
    if not payload.doc_ids:
        return BatchDeleteResponse(
            success=False,
            mode=payload.mode,
            message="doc_ids 为空",
        )

    vector = VectorStoreTool()
    vec_result = await vector.delete_by_doc_ids(payload.doc_ids)
    total_chunk_deleted = int(vec_result.get("deleted_chunk_count", 0) or 0)

    # 元数据批量
    if payload.mode == "soft":
        affected_meta = await KB_METADATA_SERVICE.mark_deleted(payload.doc_ids)
    else:
        affected_meta = await KB_METADATA_SERVICE.hard_delete_many(payload.doc_ids)

    # 联动清理结构化表数据（best-effort，失败不阻断批量删除）
    try:
        await TABLE_REGISTRY.delete_by_doc_ids(payload.doc_ids)
    except Exception as e:
        logger.warning(f"[KB批量删除] 结构化表清理失败 doc_ids={payload.doc_ids} err={e}")

    # 逐条回填详情（轻量，不重复查 DB；chunk_level 的精确删除数以总量为准）
    details: List[BatchDeleteItemResult] = []
    success_count = 0
    failed_count = 0
    for doc_id in payload.doc_ids:
        ok = bool(vec_result.get("deleted", True))
        if ok:
            success_count += 1
        else:
            failed_count += 1
        details.append(BatchDeleteItemResult(
            doc_id=doc_id,
            success=ok,
            deleted_chunk_count=0,  # 单条精确值 Chroma 不返回，用总量即可
            deleted_metadata=(affected_meta > 0),
            reason="" if ok else vec_result.get("reason", "向量库删除失败"),
        ))

    return BatchDeleteResponse(
        success=bool(vec_result.get("deleted", False)),
        mode=payload.mode,
        total=len(payload.doc_ids),
        success_count=success_count,
        failed_count=failed_count,
        deleted_chunk_count=total_chunk_deleted,
        details=details,
        message=(
            f"批量删除完成(mode={payload.mode})：成功 {success_count}/{len(payload.doc_ids)}，"
            f"清理向量 chunk 共 {total_chunk_deleted} 条；元数据 affected={affected_meta}"
        ),
    )


# ---------------------------------------------------------------------------
# 5. 文档列表（可选，对账/排障）
# ---------------------------------------------------------------------------

async def list_documents(
    kb_id: str = None,
    status: str = None,
    keyword: str = None,
    page: int = 1,
    page_size: int = 50,
) -> DocumentListResponse:
    result = await KB_METADATA_SERVICE.list_documents(
        kb_id=kb_id,
        status=status,
        keyword=keyword,
        page=page,
        page_size=page_size,
    )
    return DocumentListResponse(
        items=result.get("items", []),
        total=int(result.get("total", 0) or 0),
        page=result.get("page", page),
        page_size=result.get("page_size", page_size),
    )
