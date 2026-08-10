import httpx
from typing import Any

from pydantic import BaseModel

from app.config.settings import settings
from app.utils.logger_handle import logger


class response_type(BaseModel):
    """Java 后端统一响应格式：{code, msg, data, success}"""
    code: int = 200
    msg: str = None
    data: Any = None
    success: bool = True


def response_handler(response: response_type) -> dict:
    """统一处理 Java 后端响应（code/msg/data/success 格式）：
    - 按 code 打日志（成功 / 鉴权失败 / 参数校验失败 / 其他业务异常）
    - 归一化为 dict 返回，调用方用 response.get("success") / response["data"] 消费
    - 不抛异常：回调类调用绝不干扰 Agent 主流程
    """
    if response.code == 200:
        logger.info(f"[业务状态回调] 请求成功 code={response.code} msg={response.msg}")
    elif response.code == 401 or response.code == 403:
        logger.warning(f"[业务状态回调] 鉴权失败 code={response.code} msg={response.msg}")
    elif response.code == 422:
        logger.warning(f"[业务状态回调] 参数校验失败 code={response.code} msg={response.msg}")
    else:
        logger.warning(f"[业务状态回调] 业务异常 code={response.code} msg={response.msg}")
    return {
        "code": response.code,
        "msg": response.msg,
        "data": response.data,
        "success": response.code == 200,
    }


class JavaApiClient:
    """
    Java 后端 API 客户端
    - 统一鉴权（内部服务 token）
    - 异步 httpx（不阻塞事件循环）
    - 单例复用连接池
    - 统一响应处理（response_handler 归一化 {code, msg, data, success}）
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self) -> httpx.AsyncClient:
        if not hasattr(self, "_client") or self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.java_api_base_url,
                headers={
                    "Content-Type": "application/json",
                    "X-Service-Key": settings.java_internal_token,
                    "X-User-Id": "orchard-agent",
                },
                timeout=10.0,  # 10 秒超时
                follow_redirects=True,
            )
        return self._client

    async def post(self, path: str, json: dict = None, **kwargs) -> dict:
        resp = await self._get_client().post(path, json=json, **kwargs)
        resp.raise_for_status()
        return response_handler(response_type(**resp.json()))

    async def get(self, path: str, params: dict = None, **kwargs) -> dict:
        resp = await self._get_client().get(path, params=params, **kwargs)
        resp.raise_for_status()
        return response_handler(response_type(**resp.json()))

    async def put(self, path: str, json: dict = None, **kwargs) -> dict:
        resp = await self._get_client().put(path, json=json, **kwargs)
        resp.raise_for_status()
        return response_handler(response_type(**resp.json()))

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


API = JavaApiClient()
