"""
会话记忆存储（MongoDB，分 collection）

1. kb_conversations —— 短期窗口缓存（TTL 10 分钟）
   - 仅支撑同会话内的重试 / 续传 / Java 未传历史时的兜底
   - 短期过期的设计目的：删除一致性靠"缓存必过期"自然收敛，
     不再需要 Java 删除时回调同步（权威源在 Java MySQL）
2. kb_conversation_summaries —— 近期摘要（TTL 7 天）
   - 超出滑动窗口的早期历史压缩，由 Agent 用 LLM 增量滚动生成
   - 独立 collection 的原因：MongoDB TTL 按整个文档过期，
     10 分钟窗口与 7 天摘要无法共存于同一文档
   - 附带 summarized_upto_seq 游标：标记已总结到的消息序号，避免重复提炼
3. kb_conversation_logs —— 本地全量消息日志（TTL 30 天）
   - 滚动摘要的原料：Java 权威源只传最近 ≤10 轮，超窗的早期对话只有
     本地自持日志才拿得到，窗口滑动前从这里取待提炼消息
   - 每条消息带自增 seq（Mongo $inc 原子分配），作为摘要游标
"""
import asyncio
import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ReturnDocument

from app.config.settings import settings

# 短期窗口缓存 TTL（分钟）
WINDOW_TTL_MINUTES = 10
# 摘要 TTL（天）
SUMMARY_TTL_DAYS = 7
# 本地消息日志 TTL（天）：滚动摘要的原料，保留期覆盖长任务周期
LOG_TTL_DAYS = 30
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
        self._client = MongoClient(settings.mongodb_url, **settings.mongodb_conn_kwargs)
        db = self._client[settings.mongodb_db_name]
        self._window_collection = db["kb_conversations"]
        self._summary_collection = db["kb_conversation_summaries"]
        self._log_collection = db["kb_conversation_logs"]
        self._window_collection.create_index("thread_id", unique=True, background=True)
        self._window_collection.create_index("expire_at", expireAfterSeconds=0, background=True)
        self._summary_collection.create_index("thread_id", unique=True, background=True)
        self._summary_collection.create_index("expire_at", expireAfterSeconds=0, background=True)
        self._log_collection.create_index("thread_id", unique=True, background=True)
        self._log_collection.create_index("expire_at", expireAfterSeconds=0, background=True)

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

    def _get_summary_meta_sync(self, thread_id: str) -> Optional[dict]:
        doc = self._summary_collection.find_one({"thread_id": thread_id}, {"_id": 0})
        if not doc:
            return None
        return {
            "summary": doc.get("summary"),
            "summarized_upto_seq": doc.get("summarized_upto_seq"),
        }

    def _get_summary_sync(self, thread_id: str) -> Optional[str]:
        doc = self._summary_collection.find_one({"thread_id": thread_id}, {"_id": 0})
        return doc.get("summary") if doc else None

    def _update_summary_sync(self, thread_id: str, summary: str, summarized_upto_seq: int = None):
        now = datetime.datetime.utcnow()
        expire_at = now + datetime.timedelta(days=SUMMARY_TTL_DAYS)
        update = {"$set": {"summary": summary, "updated_at": now, "expire_at": expire_at}}
        if summarized_upto_seq is not None:
            update["$set"]["summarized_upto_seq"] = summarized_upto_seq
        self._summary_collection.update_one(
            {"thread_id": thread_id},
            update,
            upsert=True,
        )

    # ---------------- 本地全量消息日志（同步底层，滚动摘要原料） ----------------

    def _append_log_sync(self, thread_id: str, msg: dict):
        """追加一条带自增 seq 的消息。seq 由 $inc 原子分配，作为摘要游标。"""
        now = datetime.datetime.utcnow()
        expire_at = now + datetime.timedelta(days=LOG_TTL_DAYS)
        doc = self._log_collection.find_one_and_update(
            {"thread_id": thread_id},
            {
                "$setOnInsert": {"thread_id": thread_id, "expire_at": expire_at},
                "$inc": {"seq_counter": 1},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        msg["seq"] = doc["seq_counter"]
        self._log_collection.update_one(
            {"thread_id": thread_id},
            {"$push": {"messages": msg}, "$set": {"updated_at": now, "expire_at": expire_at}},
        )

    def _get_logs_sync(self, thread_id: str) -> List[Dict[str, Any]]:
        doc = self._log_collection.find_one({"thread_id": thread_id}, {"_id": 0})
        return doc.get("messages", []) if doc else []

    def _delete_log_messages_sync(self, thread_id: str, msg_ids: List[str]):
        """
        按 Java 消息 id 删除日志消息（user 消息 + 紧随其后的 assistant 回答，配对删除）。
        返回 (matched_count, summary_invalidated)：
        - matched_count：匹配删除的 user 消息数（Java 可对账）
        - summary_invalidated：被删消息是否落在已总结区间（seq ≤ 游标），
          若是则摘要含被删内容，需要调用方作废摘要，下次滚动全量重建
        """
        doc = self._log_collection.find_one({"thread_id": thread_id}, {"_id": 0})
        if not doc:
            return 0, False
        msgs = doc.get("messages", [])
        ids = set(msg_ids)
        to_delete = set()
        matched = 0
        min_deleted_seq = None
        for i, m in enumerate(msgs):
            if m.get("role") != "user":
                continue
            if m.get("msg_id") not in ids:
                continue
            matched += 1
            to_delete.add(i)
            seq = m.get("seq")
            if min_deleted_seq is None or seq < min_deleted_seq:
                min_deleted_seq = seq
            # 配对：删除紧随其后的 assistant 回答（同轮）
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                to_delete.add(i + 1)
        if not to_delete:
            return 0, False
        keep = [m for i, m in enumerate(msgs) if i not in to_delete]
        self._log_collection.update_one(
            {"thread_id": thread_id},
            {"$set": {"messages": keep}},
        )
        # 摘要失效判定：被删消息中最小 seq ≤ summarized_upto_seq → 已总结过该段
        summary_doc = self._summary_collection.find_one({"thread_id": thread_id}, {"_id": 0})
        upto = (summary_doc or {}).get("summarized_upto_seq")
        invalidated = upto is not None and min_deleted_seq is not None and min_deleted_seq <= upto
        return matched, invalidated

    def _clear_all_sync(self, thread_id: str):
        """清空该会话的全部本地记忆缓存（短期窗口 / 摘要 / 全量日志）"""
        self._window_collection.delete_one({"thread_id": thread_id})
        self._summary_collection.delete_one({"thread_id": thread_id})
        self._log_collection.delete_one({"thread_id": thread_id})

    def _reset_summary_sync(self, thread_id: str):
        """作废摘要：删除摘要文档（游标一并清除），下次滚动从剩余日志全量重建"""
        self._summary_collection.delete_one({"thread_id": thread_id})

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

    async def get_summary_meta(self, thread_id: str):
        """获取该会话的摘要及摘要游标，返回 (summary, summarized_upto_seq)。"""
        self._ensure_init()
        meta = await asyncio.to_thread(self._get_summary_meta_sync, str(thread_id))
        if not meta:
            return None, None
        return meta.get("summary"), meta.get("summarized_upto_seq")

    async def update_summary(self, thread_id: str, summary: str, summarized_upto_seq: int = None) -> None:
        """写入/覆盖该会话摘要；summarized_upto_seq 为滚动摘要游标（已总结到的消息序号）。"""
        self._ensure_init()
        await asyncio.to_thread(
            self._update_summary_sync, str(thread_id), summary, summarized_upto_seq
        )

    # ---------------- 本地全量消息日志 API（滚动摘要原料） ----------------

    async def append_log(self, thread_id: str, role: str, content: str, msg_id: Optional[str] = None) -> None:
        """把一轮消息（user/assistant）追加到本地全量日志，自动分配自增 seq。
        msg_id 为 Java 端消息 ID（user 消息传入），用于业务端删除消息时的本地定位。"""
        self._ensure_init()
        msg = {
            "role": role,
            "content": content,
            "ts": int(datetime.datetime.utcnow().timestamp() * 1000),
        }
        if msg_id:
            msg["msg_id"] = msg_id
        await asyncio.to_thread(self._append_log_sync, str(thread_id), msg)

    async def get_logs(self, thread_id: str) -> List[Dict[str, Any]]:
        """获取本地全量消息日志（含 seq，按插入顺序）。"""
        self._ensure_init()
        return await asyncio.to_thread(self._get_logs_sync, str(thread_id))

    async def delete_log_messages(self, thread_id: str, message_ids: List[str]):
        """按 Java 消息 id 删除日志消息，返回 (matched_count, summary_invalidated)。"""
        self._ensure_init()
        return await asyncio.to_thread(
            self._delete_log_messages_sync, str(thread_id), list(message_ids)
        )

    async def clear_all(self, thread_id: str) -> None:
        """清空该会话的全部本地记忆缓存（业务端删除会话时回调）"""
        self._ensure_init()
        await asyncio.to_thread(self._clear_all_sync, str(thread_id))

    async def reset_summary(self, thread_id: str) -> None:
        """作废摘要（删除已总结区间内的消息后，下次滚动全量重建）"""
        self._ensure_init()
        await asyncio.to_thread(self._reset_summary_sync, str(thread_id))


chat_memory = ChatMemoryStore()
