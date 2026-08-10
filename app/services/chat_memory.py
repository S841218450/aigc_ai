"""
会话记忆存储（MongoDB，分两个 collection）

1. kb_conversations —— 短期窗口缓存（TTL 10 分钟）
   - 仅支撑同会话内的重试 / 续传 / Java 未传历史时的兜底
   - 短期过期的设计目的：删除一致性靠"缓存必过期"自然收敛，
     不再需要 Java 删除时回调同步（权威源在 Java MySQL）
2. kb_conversation_summaries —— 近期摘要（TTL 7 天）
   - 超出滑动窗口的早期历史压缩，由 Agent 用 LLM 增量滚动生成
   - 独立 collection 的原因：MongoDB TTL 按整个文档过期，
     10 分钟窗口与 7 天摘要无法共存于同一文档
"""
import asyncio
import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

from app.config.settings import settings

# 短期窗口缓存 TTL（分钟）
WINDOW_TTL_MINUTES = 10
# 摘要 TTL（天）
SUMMARY_TTL_DAYS = 7
# 短期缓存保留的最大轮数（每轮 = 1 user + 1 assistant），超出裁剪
MAX_HISTORY_TURNS = 20


class ChatMemoryStore:
    _instance = None
    _client: MongoClient = None
    _window_collection = None
    _summary_collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_init(self):
        if self._window_collection is not None:
            return
        self._client = MongoClient(settings.mongodb_url)
        db = self._client[settings.mongodb_db_name]
        self._window_collection = db["kb_conversations"]
        self._summary_collection = db["kb_conversation_summaries"]
        self._window_collection.create_index("thread_id", unique=True, background=True)
        self._window_collection.create_index("expire_at", expireAfterSeconds=0, background=True)
        self._summary_collection.create_index("thread_id", unique=True, background=True)
        self._summary_collection.create_index("expire_at", expireAfterSeconds=0, background=True)

    # ---------------- 短期窗口缓存（同步底层） ----------------

    def _append_sync(self, thread_id: str, msg: dict):
        now = datetime.datetime.utcnow()
        expire_at = now + datetime.timedelta(minutes=WINDOW_TTL_MINUTES)
        doc = self._window_collection.find_one({"thread_id": thread_id})
        if doc:
            msgs = doc.get("messages", [])
            msgs.append(msg)
            max_msgs = MAX_HISTORY_TURNS * 2
            if len(msgs) > max_msgs:
                msgs = msgs[-max_msgs:]
            self._window_collection.update_one(
                {"thread_id": thread_id},
                {"$set": {"messages": msgs, "updated_at": now, "expire_at": expire_at}},
            )
        else:
            self._window_collection.insert_one({
                "thread_id": thread_id,
                "messages": [msg],
                "updated_at": now,
                "expire_at": expire_at,
            })

    def _get_sync(self, thread_id: str) -> Optional[dict]:
        return self._window_collection.find_one({"thread_id": thread_id}, {"_id": 0})

    def _clear_sync(self, thread_id: str):
        self._window_collection.delete_one({"thread_id": thread_id})

    # ---------------- 摘要存储（同步底层） ----------------

    def _get_summary_sync(self, thread_id: str) -> Optional[str]:
        doc = self._summary_collection.find_one({"thread_id": thread_id}, {"_id": 0})
        return doc.get("summary") if doc else None

    def _update_summary_sync(self, thread_id: str, summary: str):
        now = datetime.datetime.utcnow()
        expire_at = now + datetime.timedelta(days=SUMMARY_TTL_DAYS)
        self._summary_collection.update_one(
            {"thread_id": thread_id},
            {"$set": {"summary": summary, "updated_at": now, "expire_at": expire_at}},
            upsert=True,
        )

    # ---------------- 对外异步 API ----------------

    async def append(
        self,
        thread_id: str,
        role: str,
        content: str,
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """追加一条短期缓存消息。role: user / assistant。params 仅 user 消息携带（重试用）。"""
        self._ensure_init()
        msg: Dict[str, Any] = {
            "role": role,
            "content": content,
            "ts": int(datetime.datetime.utcnow().timestamp() * 1000),
        }
        if role == "user":
            if params:
                msg["params"] = params
            if user_id:
                msg["userId"] = user_id
        await asyncio.to_thread(self._append_sync, str(thread_id), msg)

    async def get_messages(self, thread_id: str, max_turns: int = 10) -> List[Dict[str, Any]]:
        """
        获取短期缓存中最近 max_turns 轮（Java 未传 chat_history 时的兜底）。
        返回 [{"role": ..., "content": ...}, ...]
        """
        self._ensure_init()
        doc = await asyncio.to_thread(self._get_sync, str(thread_id))
        if not doc:
            return []
        msgs = doc.get("messages", [])
        max_msgs = max_turns * 2
        if len(msgs) > max_msgs:
            msgs = msgs[-max_msgs:]
        return [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in msgs if m.get("content")
        ]

    async def get_last_user_query(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """获取短期缓存中最后一条用户消息（供重试接口）"""
        self._ensure_init()
        doc = await asyncio.to_thread(self._get_sync, str(thread_id))
        if not doc:
            return None
        for m in reversed(doc.get("messages", [])):
            if m.get("role") == "user":
                return {
                    "content": m.get("content", ""),
                    "params": m.get("params") or {},
                    "userId": m.get("userId") or "",
                }
        return None

    async def clear(self, thread_id: str) -> None:
        """清空指定会话的短期缓存"""
        self._ensure_init()
        await asyncio.to_thread(self._clear_sync, str(thread_id))

    # ---------------- 摘要 API ----------------

    async def get_summary(self, thread_id: str) -> Optional[str]:
        """获取该会话的近期摘要（无则 None）"""
        self._ensure_init()
        return await asyncio.to_thread(self._get_summary_sync, str(thread_id))

    async def update_summary(self, thread_id: str, summary: str) -> None:
        """写入/覆盖该会话摘要"""
        self._ensure_init()
        await asyncio.to_thread(self._update_summary_sync, str(thread_id), summary)


chat_memory = ChatMemoryStore()
