import asyncio
import datetime
from typing import Any, List, Optional
from pymongo import MongoClient, ASCENDING

from app.config.settings import settings


class EventStore:
    """
    SSE 事件持久化存储（MongoDB）
    - 以 thread_id + seq_id 为索引
    - 支持断点续传：根据 Last-Event-ID 查询遗漏事件
    - 自动过期清理（TTL 24h）
    - 所有 pymongo 操作通过 asyncio.to_thread 异步化，不阻塞事件循环
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
        self._client = MongoClient(settings.mongodb_url)
        db = self._client[settings.mongodb_db_name]
        self._collection = db["sse_events"]

        # 索引：按 thread_id + seq_id 查询 + TTL 自动过期
        self._collection.create_index(
            [("thread_id", ASCENDING), ("seq_id", ASCENDING)],
            background=True,
        )
        self._collection.create_index(
            "expire_at",
            expireAfterSeconds=0,
            background=True,
        )

    def _save_sync(self, doc: dict):
        self._collection.insert_one(doc)

    def _find_after_sync(self, thread_id: str, after_seq_id: int) -> List[dict]:
        cursor = self._collection.find(
            {"thread_id": thread_id, "seq_id": {"$gt": after_seq_id}},
            {"_id": 0},
        ).sort("seq_id", ASCENDING)
        return list(cursor)

    def _find_last_seq_sync(self, thread_id: str) -> Optional[int]:
        doc = self._collection.find_one(
            {"thread_id": thread_id},
            sort=[("seq_id", -1)],
            projection={"seq_id": 1, "_id": 0},
        )
        return doc["seq_id"] if doc else None

    def _clear_sync(self, thread_id: str):
        self._collection.delete_many({"thread_id": thread_id})

    async def save(self, event):
        """保存一个 SSE 事件到 MongoDB（异步，不阻塞事件循环）"""
        self._ensure_init()
        doc = {
            "thread_id": str(event.thread_id),
            "seq_id": event.seq_id,
            "type": event.type,
            "status": event.status,
            "data": event.data,
            "timestamp": event.timestamp,
            "expire_at": datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        }
        await asyncio.to_thread(self._save_sync, doc)

    async def get_events_after(self, thread_id: str, after_seq_id: int) -> List[dict]:
        """获取指定 thread_id 中 seq_id > after_seq_id 的所有事件（用于断点续传）"""
        self._ensure_init()
        return await asyncio.to_thread(self._find_after_sync, str(thread_id), after_seq_id)

    async def get_last_seq_id(self, thread_id: str) -> Optional[int]:
        """获取指定 thread_id 的最大 seq_id"""
        self._ensure_init()
        return await asyncio.to_thread(self._find_last_seq_sync, str(thread_id))

    async def clear_thread(self, thread_id: str):
        """清理指定 thread_id 的所有事件"""
        self._ensure_init()
        await asyncio.to_thread(self._clear_sync, str(thread_id))


event_store = EventStore()
