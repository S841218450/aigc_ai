"""
知识库文档元数据服务（MongoDB）
================================
- 单一入口管理文档的元数据（来源、chunk 数、处理状态、失败原因）
- 与向量库（Chroma）的 doc_id 完全对齐（以 Java 端生成的 doc_id 为主键）
- 所有 DB 操作通过 asyncio.to_thread 异步化，不阻塞事件循环

分层职责：
- 文件下载/解析工具：app/tools/common/file_download.py
- 文档处理编排（入库/删除/列表）：app/services/knowledge_base_doc_service.py
- 问答会话编排（记忆上下文/graph）：app/services/knowledge_base_chat_service.py
"""
import asyncio
import datetime
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING, DESCENDING

from app.api.java_client import API
from app.config.settings import settings
from app.utils.logger_handle import logger


# ---------------------------------------------------------------------------
# 1. MongoDB 文档元数据 CRUD
# ---------------------------------------------------------------------------

class KnowledgeBaseDocMetadataService:
    """
    kb_documents collection:
      {
        "_id": "<mongo auto>",
        "doc_id": "<Java端唯一ID，主键唯一索引>",
        "kb_id": "<预留，知识库ID，默认 'default'>",
        "doc_name": "产品白皮书V2.pdf",
        "file_url": "https://cos.xxx/xxx.pdf",
        "file_md5": "a1b2c3...",
        "file_size": 2097152,
        "file_type": "pdf",
        "chunk_count": 42,
        "status": "processing" | "ready" | "failed" | "deleted",
        "fail_reason": "PDF 加密无法解析",
        "metadata": { permission, owner, tags, uploader, ... (Java 传入的扩展元数据) },
        "created_at": "ISO",
        "updated_at": "ISO",
      }
    """

    _instance = None
    _client: MongoClient = None
    _collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_init(self):
        if self._collection is not None:
            return
        self._client = MongoClient(settings.mongodb_url, **settings.mongodb_conn_kwargs)
        db = self._client[settings.mongodb_db_name]
        self._collection = db["kb_documents"]

        # 索引：doc_id 唯一，对齐 Java 端
        self._collection.create_index(
            [("doc_id", ASCENDING)],
            unique=True,
            background=True,
        )
        # 常用查询索引
        self._collection.create_index(
            [("status", ASCENDING), ("updated_at", DESCENDING)],
            background=True,
        )
        self._collection.create_index(
            [("kb_id", ASCENDING)],
            background=True,
        )

    # -------- sync 版本（给 asyncio.to_thread 用） --------

    def _create_sync(self, doc: Dict[str, Any]) -> bool:
        try:
            self._collection.replace_one(
                {"doc_id": doc["doc_id"]},
                doc,
                upsert=True,
            )
            return True
        except Exception as e:
            logger.error(f"[KB Metadata] create 失败(doc_id={doc.get('doc_id')}): {e}")
            return False

    def _get_sync(self, doc_id: str) -> Optional[Dict[str, Any]]:
        doc = self._collection.find_one(
            {"doc_id": doc_id, "status": {"$ne": "deleted"}},
            {"_id": 0},
        )
        return doc

    def _list_sync(
        self,
        *,
        kb_id: str = None,
        status: str = None,
        keyword: str = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[Dict[str, Any]], int]:
        query: Dict[str, Any] = {"status": {"$ne": "deleted"}}
        if kb_id:
            query["kb_id"] = kb_id
        if status:
            query["status"] = status
        if keyword:
            query["doc_name"] = {"$regex": keyword, "$options": "i"}

        total = self._collection.count_documents(query)
        skip = max(0, (page - 1) * page_size)
        cursor = (
            self._collection.find(query, {"_id": 0})
            .sort("updated_at", DESCENDING)
            .skip(skip)
            .limit(page_size)
        )
        return list(cursor), total

    def _update_status_sync(
        self,
        doc_id: str,
        status: str,
        *,
        chunk_count: int = None,
        fail_reason: str = None,
        extra_update: Dict[str, Any] = None,
    ) -> bool:
        update: Dict[str, Any] = {
            "status": status,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        if chunk_count is not None:
            update["chunk_count"] = chunk_count
        if fail_reason:
            update["fail_reason"] = fail_reason
        if extra_update:
            update.update(extra_update)
        result = self._collection.update_one(
            {"doc_id": doc_id},
            {"$set": update},
        )
        return result.modified_count > 0

    def _hard_delete_sync(self, doc_id: str) -> int:
        result = self._collection.delete_one({"doc_id": doc_id})
        return result.deleted_count

    def _hard_delete_many_sync(self, doc_ids: List[str]) -> int:
        result = self._collection.delete_many({"doc_id": {"$in": doc_ids}})
        return result.deleted_count

    def _mark_deleted_sync(self, doc_ids: List[str]) -> int:
        now = datetime.datetime.now().isoformat()
        result = self._collection.update_many(
            {"doc_id": {"$in": doc_ids}},
            {"$set": {"status": "deleted", "updated_at": now}},
        )
        return result.modified_count

    # -------- async 对外接口 --------

    async def create_document(
        self,
        *,
        doc_id: str,
        doc_name: str,
        file_url: str = "",
        file_md5: str = "",
        file_size: int = 0,
        file_type: str = "",
        metadata: Dict[str, Any] = None,
        kb_id: str = "default",
    ) -> bool:
        """创建或 upsert 文档元数据（status=processing）"""
        self._ensure_init()
        now = datetime.datetime.now().isoformat()
        doc = {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "doc_name": doc_name,
            "file_url": file_url,
            "file_md5": file_md5,
            "file_size": file_size,
            "file_type": file_type,
            "chunk_count": 0,
            "status": "processing",
            "fail_reason": "",
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }
        return await asyncio.to_thread(self._create_sync, doc)

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_init()
        return await asyncio.to_thread(self._get_sync, doc_id)

    async def list_documents(
        self,
        *,
        kb_id: str = None,
        status: str = None,
        keyword: str = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        self._ensure_init()
        items, total = await asyncio.to_thread(
            self._list_sync,
            kb_id=kb_id,
            status=status,
            keyword=keyword,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    async def mark_ready(
        self,
        doc_id: str,
        *,
        chunk_count: int,
        extra_update: Dict[str, Any] = None,
    ) -> bool:
        """入库成功，标记文档为 ready"""
        self._ensure_init()
        return await asyncio.to_thread(
            self._update_status_sync,
            doc_id,
            "ready",
            chunk_count=chunk_count,
            extra_update=extra_update,
        )

    async def mark_failed(self, doc_id: str, reason: str) -> bool:
        """入库失败，标记 failed 并记录原因"""
        self._ensure_init()
        return await asyncio.to_thread(
            self._update_status_sync,
            doc_id,
            "failed",
            fail_reason=reason,
        )

    async def mark_deleted(self, doc_ids: List[str]) -> int:
        """软删（status=deleted），给对账/回滚保留窗口；真正物理删走 hard_delete"""
        self._ensure_init()
        return await asyncio.to_thread(self._mark_deleted_sync, doc_ids)

    async def hard_delete(self, doc_id: str) -> int:
        self._ensure_init()
        return await asyncio.to_thread(self._hard_delete_sync, doc_id)

    async def hard_delete_many(self, doc_ids: List[str]) -> int:
        self._ensure_init()
        return await asyncio.to_thread(self._hard_delete_many_sync, doc_ids)


KB_METADATA_SERVICE = KnowledgeBaseDocMetadataService()




# 修改知识库文档状态 → 回写 Java
async def change_knowledge_doc_status(
    doc_id: str,
    status: str,
    *,
    chunk_count: int = 0,
    fail_reason: str = "",
    file_size: int = 0,
) -> dict:
    """Agent 文档入库成功/失败/跳过 → 同步回写 Java 端知识库文档状态。
    响应为 Java 统一格式 {code, msg, data, success}，由 java_client.response_handler 归一化。"""
    status_map = {
        "processing": 1,  # 向量化中
        "ready": 2,       # 已完成
        "failed": 3,      # 失败
        "skipped": 2,     # 幂等跳过（已 ready）→ 已完成
    }
    body: Dict[str, Any] = {
        "id": doc_id,
        "status": status_map.get(status, 0),
        "chunkCount": int(chunk_count or 0),
        "fileSize": int(file_size or 0),
        "errorMsg": fail_reason or "",
    }

    return await API.put("/api/ai/knowledge/status", body)


# 更新消息数据 → 回写 Java
async def update_chat(id: str, answer: str, status: int = 1, errorMsg: str = "") -> dict:
    """把 Agent 生成的回答回写 Java 端消息（status: 1-完成 2-失败）。
    响应为 Java 统一格式 {code, msg, data, success}，由 java_client.response_handler 归一化。"""
    if not id:
        return {"code": 400, "msg": "id 为空", "data": None, "success": False}
    body: Dict[str, Any] = {
        "id": id,
        "status": status,#1-完成 2-失败
        "answer": answer,
        "errorMsg": errorMsg,
    }
    return await API.put("/api/ai/chat/update", body)

# 获取滚动窗口消息
async def get_chat_window(sessionId: str, ) -> dict:
    """获取滚动窗口窗口消息（success: true-完成 false-失败）。"""
    if not sessionId:
        return {"code": 400, "msg": "sessionId 为空", "data": None, "success": False}
    body: Dict[str, Any] = {
        "id": sessionId,
        "pageNum": 1,
        "pageSize": 10,#只获取十条消息
    }
    return await API.get("/api/ai/message/page", body)
