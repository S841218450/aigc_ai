from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

from app.config.settings import settings


class CheckpointerService:
    """
    MongoDB Checkpointer 单例服务
    - 以 threadId + userId 为索引持久化节点状态
    - 全局复用同一个连接
    """

    _instance = None
    _checkpointer: MongoDBSaver = None
    _client: MongoClient = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_checkpointer(self, collection_name: str = "checkpoints") -> MongoDBSaver:
        if self._checkpointer is None:
            self._client = MongoClient(settings.mongodb_url)
            db = self._client[settings.mongodb_db_name]
            collection = db[collection_name]

            # 创建索引：thread_id 查询最频繁
            collection.create_index("thread_id", background=True)
            collection.create_index([("thread_id", 1), ("user_id", 1)], background=True)

            self._checkpointer = MongoDBSaver(
                client=self._client,
                db_name=settings.mongodb_db_name,
                collection_name=collection_name,
            )
        return self._checkpointer

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._checkpointer = None


checkpointer_service = CheckpointerService()
