"""
知识库 API 路由（薄 HTTP 层）
============================
职责：
- 路由定义、HTTP 请求/响应组装（SSE、query/stop/retry、内部文档接口）
- 内部服务鉴权（X-Service-Key）
- 不做业务编排：查询走 app/services/knowledge_base_chat_service.py，
  文档入库/删除/列表走 app/services/knowledge_base_doc_service.py，
  历史解析纯函数在 app/utils/chat_history.py，文件下载/解析在 app/tools/common/file_download.py
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Path, Query, Request

from app.config.settings import settings
from app.models.schemas.knowledge_base import (
    KnowledgeBaseQueryRequest,
    KnowledgeBaseUploadRequest,
    KnowledgeBaseUploadResponse,
    DeleteResponse,
    BatchDeleteRequest,
    BatchDeleteResponse,
    DocumentListResponse,
    StopGenerationRequest,
    RetryRequest,
    ControlResponse,
    MemoryClearRequest,
    MemoryDeleteMessagesRequest,
    MemoryDeleteMessagesResponse,
    MemoryTitleRequest,
)
from app.services.chat_memory import chat_memory
from app.services.event_store import event_store
from app.services.knowledge_base_chat_service import (
    clear_session_memory,
    delete_session_messages,
    run_query_stream,
    spawn_title_task,
)
from app.services.knowledge_base_doc_service import (
    submit_ingest_document,
    delete_document,
    batch_delete_documents,
    list_documents,
)
from app.services.node_status import KNOWLEDGE_BASE_NODE_MAP, build_node_data
from app.services.sse_service import (
    graph_to_sse_events,
    replay_events,
    build_sse_response,
    task_manager,
)
from app.utils.chat_history import parse_chat_history

router = APIRouter()


def _build_node_data(node_name: str, state_update: dict) -> dict:
    """节点状态 → SSE data 包装（统一透传给 graph_to_sse_events）"""
    return build_node_data(node_name, state_update, node_map=KNOWLEDGE_BASE_NODE_MAP)


# ---------------------------------------------------------------------------
# 查询（SSE 流式）+ 停止 / 重试
# ---------------------------------------------------------------------------

@router.post("/query")
async def query_knowledge_base(request: KnowledgeBaseQueryRequest, req: Request):
    # 数据初始化
    thread_id = request.threadId or str(uuid.uuid4())
    last_event_id = req.headers.get("Last-Event-ID")
    last_seq_id = int(last_event_id) if last_event_id else None

    # 知识库过滤
    params = request.params or {}
    params_filter = dict(params.get("filter") or {})
    if request.kb_id or params.get("kb_id"):
        params_filter["kb_id"] = request.kb_id or params.get("kb_id") or "default"
    if request.filter_folder_ids:
        if len(request.filter_folder_ids) == 1:
            params_filter["folder_id"] = request.filter_folder_ids[0]
        else:
            params_filter["folder_id"] = {"$in": request.filter_folder_ids}
    if request.filter_doc_ids:
        if len(request.filter_doc_ids) == 1:
            params_filter["doc_id"] = request.filter_doc_ids[0]
        else:
            params_filter["doc_id"] = {"$in": request.filter_doc_ids}
    if params_filter:
        params["filter"] = params_filter

    # 滑动窗口：Java 从 MySQL 现取最近 ≤10 轮传入（权威源）；未传时 Agent 用短期缓存兜底
    message_history = None
    if request.chat_history:
        message_history = parse_chat_history(request.chat_history)
        # 去重：Java 会把当前进行中消息（question 与当前问题相同、answer 为空）放进
        # chat_history，parse 后已保留其 user 消息，这里剔除，避免与 query 重复注入
        if message_history and message_history[-1]["role"] == "user" and message_history[-1]["content"] == request.question:
            message_history.pop()
    message_id = request.messageId or request.threadId or ""  # 优先新字段 messageId；兼容旧约定回退 threadId

    sse, wrapped = await run_query_stream(
        thread_id, request.question, params,
        userId=request.userId or "",
        kb_id=request.kb_id or params.get("kb_id"),
        filter_folder_ids=request.filter_folder_ids,
        filter_doc_ids=request.filter_doc_ids,
        attachments=[a.model_dump() for a in request.attachments] if request.attachments else None,
        append_user=True,
        message_history=message_history,
        message_id=message_id,
    )

    async def event_generator():
        if last_seq_id is not None:
            async for chunk in replay_events(str(thread_id), last_seq_id, sse, event_store):
                yield chunk
        async for chunk in sse.stream(graph_to_sse_events(
            wrapped, sse, thread_id,
            node_map=KNOWLEDGE_BASE_NODE_MAP,
            build_node_data=_build_node_data,
        )):
            yield chunk

    return build_sse_response(event_generator())


@router.post("/stop", response_model=ControlResponse)
async def stop_generation(request: StopGenerationRequest):
    """停止指定会话正在进行的生成任务（幂等：无活跃任务时也返回 success=False，不报错）"""
    thread_id = request.threadId
    if not thread_id:
        raise HTTPException(status_code=400, detail="threadId 不能为空")
    cancelled = task_manager.cancel(thread_id)
    return ControlResponse(
        success=cancelled,
        threadId=thread_id,
        message="已停止生成" if cancelled else "没有正在进行的生成任务（可能已结束或超时）",
    )


@router.post("/retry", summary="重试该会话最后一次用户查询（SSE 流式返回）")
async def retry_query(request: RetryRequest, req: Request):
    """重新执行该会话最后一次用户查询。不重复写入 user 历史消息。"""
    thread_id = request.threadId
    if not thread_id:
        raise HTTPException(status_code=400, detail="threadId 不能为空")
    last = await chat_memory.get_last_user_query(thread_id)
    if not last or not last.get("content"):
        raise HTTPException(status_code=404, detail="该会话没有可重试的历史消息")

    params = last.get("params") or {}
    sse, wrapped = await run_query_stream(
        thread_id, last["content"], params,
        userId=last.get("userId") or "",
        kb_id=params.get("kb_id"),
        append_user=False,
    )

    async def event_generator():
        async for chunk in sse.stream(graph_to_sse_events(
            wrapped, sse, thread_id,
            node_map=KNOWLEDGE_BASE_NODE_MAP,
            build_node_data=_build_node_data,
        )):
            yield chunk

    return build_sse_response(event_generator())


# ---------------------------------------------------------------------------
# 内部接口：文档入库 / 删除 / 列表（供 Java 业务端回调，X-Service-Key 鉴权）
# ---------------------------------------------------------------------------

async def require_internal_service_key(x_service_key: Optional[str] = Header(None, alias="X-Service-Key")):
    """
    内部服务调用鉴权：请求头必须携带有效的 X-Service-Key（与 settings.java_internal_token 一致）
    供 Java 端调用 Agent 的文档管理接口时使用。
    """
    if not settings.java_internal_token:
        # 环境变量未配置时放行（本地开发），但打一条警告日志
        import warnings
        warnings.warn("⚠️  JAVA_INTERNAL_TOKEN 未配置，内部文档接口处于无鉴权放行状态，生产环境必须配置！")
        return
    if not x_service_key or x_service_key != settings.java_internal_token:
        raise HTTPException(status_code=401, detail="无效或缺失的内部服务鉴权令牌 (X-Service-Key)")


_internal_router_depends = [Depends(require_internal_service_key)]


# ----------------------------- 1. 文档入库 -----------------------------

@router.post(
    "/internal/documents",
    response_model=KnowledgeBaseUploadResponse,
    dependencies=_internal_router_depends,
    summary="[内部] 上传文档并入库向量（提交即返回 200，后台异步入库后回调状态接口）",
)
async def upload_document(payload: KnowledgeBaseUploadRequest):
    """
    提交即返回 200：快速校验后立即响应，后台异步执行分割/向量入库/结构化登记，
    完成后回调 PUT /api/ai/knowledge/status 更新 Java 端状态。
    避免 Excel 解析 / 大文件向量化耗时导致 Java 端同步超时误判上传失败。
    """
    return await submit_ingest_document(payload)


# ----------------------------- 2. 单文档删除 -----------------------------

@router.delete(
    "/internal/documents/{doc_id}",
    response_model=DeleteResponse,
    dependencies=_internal_router_depends,
    summary="[内部] 按 doc_id 物理删除向量库及对应的元数据",
)
async def remove_document(
    doc_id: str = Path(..., description="Java 端分配的唯一文档 ID"),
):
    """Java 端在删除文件成功后回调此接口，物理删除向量 chunk 和元数据，避免「幽灵 chunk」"""
    return await delete_document(doc_id)


# ----------------------------- 3. 批量删除 -----------------------------

@router.post(
    "/internal/documents/batch_delete",
    response_model=BatchDeleteResponse,
    dependencies=_internal_router_depends,
    summary="[内部] 批量删除（建议单次 <=100 个 doc_ids）",
)
async def batch_remove_documents(payload: BatchDeleteRequest):
    """批量删除：向量库单次 where $in + 元数据按 mode 批量处理，逐条回填 details 供对账"""
    return await batch_delete_documents(payload)


# ----------------------------- 4. 文档列表（可选，对账/排障） -----------------------------

@router.get(
    "/internal/documents",
    response_model=DocumentListResponse,
    dependencies=_internal_router_depends,
    summary="[内部] 文档列表（供排障/对账，不要对前端开放）",
)
async def query_documents(
    kb_id: Optional[str] = Query(None, description="知识库 ID，默认查全部"),
    status: Optional[str] = Query(None, description="processing/ready/failed/deleted"),
    keyword: Optional[str] = Query(None, description="按 doc_name 模糊搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    return await list_documents(
        kb_id=kb_id,
        status=status,
        keyword=keyword,
        page=page,
        page_size=page_size,
    )


# ----------------------------- 5. 记忆同步（会话/消息删除回调） -----------------------------

@router.post(
    "/internal/memory/clear",
    response_model=ControlResponse,
    dependencies=_internal_router_depends,
    summary="[内部] 删除会话时回调：清空 Agent 本地该会话全部记忆缓存",
)
async def clear_session_memory_ep(payload: MemoryClearRequest):
    """业务端删除整个会话（含所有消息）后回调，清理 Agent 本地短期窗口/摘要/全量日志，
    防止已删除内容通过本地缓存回灌上下文。幂等。"""
    await clear_session_memory(payload.threadId)
    return ControlResponse(success=True, threadId=payload.threadId, message="会话记忆缓存已清理")


@router.post(
    "/internal/memory/messages/delete",
    response_model=MemoryDeleteMessagesResponse,
    dependencies=_internal_router_depends,
    summary="[内部] 删除某段消息时回调：从 Agent 本地日志删除对应消息（含其后的回答）",
)
async def delete_session_messages_ep(payload: MemoryDeleteMessagesRequest):
    """业务端删除一段对话（用户消息 + 其回答）后回调，同步清理 Agent 本地日志。
    若被删消息已进入摘要，摘要作废并于下次请求时基于剩余日志重建。"""
    result = await delete_session_messages(payload.threadId, payload.messageIds)
    return MemoryDeleteMessagesResponse(
        success=True,
        threadId=payload.threadId,
        matched=result["matched"],
        summary_reset=result["summary_reset"],
        message=f"已匹配删除 {result['matched']} 条消息",
    )


@router.post(
    "/internal/memory/title",
    response_model=ControlResponse,
    dependencies=_internal_router_depends,
    summary="[内部] 新建会话时回调：异步生成会话标题并回调 Java 保存",
)
async def generate_title_ep(payload: MemoryTitleRequest):
    """Java 新建会话时携带第一条消息调用本接口。

    标题生成是异步的（LLM 提炼 → 回调 Java 保存），本接口**立即返回成功**，
    不阻塞用户的第一条对话；生成/保存失败仅记日志，不影响会话。
    """
    spawn_title_task(payload.threadId, payload.question)
    return ControlResponse(
        success=True,
        threadId=payload.threadId,
        message="标题生成任务已受理（异步执行）",
    )
