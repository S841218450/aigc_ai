import logging
import os
import time
import json
import traceback
import warnings
from datetime import datetime
from typing import Optional, Set

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response, JSONResponse
from starlette.concurrency import iterate_in_threadpool

from app.utils.path_tool import get_absolute_path

LOG_PATH = get_absolute_path("logs")

os.makedirs(LOG_PATH, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# 需要脱敏的请求字段（不区分大小写）
_SENSITIVE_FIELDS: Set[str] = {
    "password", "pwd", "secret", "token", "access_token", "refresh_token",
    "authorization", "api_key", "apikey", "x-service-key", "x_service_key"
}

# 请求体日志最大长度（避免打印超大 payload）
_MAX_BODY_LOG_LEN = 4000

# 已被 setup 过的 FastAPI app id，避免重复挂载中间件
_SETUP_APPS: Set[int] = set()


def _mask_sensitive(data: dict) -> dict:
    """递归脱敏字典中的敏感字段"""
    if not isinstance(data, dict):
        return data
    result = {}
    for k, v in data.items():
        if isinstance(k, str) and k.lower() in _SENSITIVE_FIELDS:
            result[k] = "***MASKED***"
        elif isinstance(v, dict):
            result[k] = _mask_sensitive(v)
        elif isinstance(v, list):
            result[k] = [_mask_sensitive(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


def _truncate(text: str, max_len: int = _MAX_BODY_LOG_LEN) -> str:
    if not text or len(text) <= max_len:
        return text or ""
    return text[:max_len] + f"...(truncated, total {len(text)} chars)"


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _has_usable_handlers(lg: logging.Logger) -> bool:
    """判断 logger 自身是否已经有真正可用的 handler（不包括 placeholder NullHandler）"""
    for h in lg.handlers:
        if not isinstance(h, logging.NullHandler):
            return True
    return False


def _find_ancestor_with_handlers(name: str) -> Optional[logging.Logger]:
    """沿 logger 名称层级向上找，看是否有祖先 logger 已经挂了 handler。"""
    parts = name.split(".")
    # 从父级开始往上走，不包括自身
    for i in range(len(parts) - 1, 0, -1):
        ancestor_name = ".".join(parts[:i])
        ancestor = logging.getLogger(ancestor_name)
        if _has_usable_handlers(ancestor):
            return ancestor
    # root logger
    root = logging.getLogger()
    if _has_usable_handlers(root):
        return root
    return None


def get_logger(
        name: str = 'agent',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file: Optional[str] = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 已经有 handler：直接返回（避免重复挂）
    if _has_usable_handlers(logger):
        return logger

    # 子 logger（如 agent.access）：如果父级（如 agent）已有 handler，
    # 开启 propagate 让父级输出即可，不重复挂 handler 避免重复打印。
    if "." in name:
        ancestor = _find_ancestor_with_handlers(name)
        if ancestor is not None:
            logger.propagate = True
            return logger

    # 顶级 logger（如 agent / agent.access 且没有可用父级）：挂 handler 并禁止 propagate 到 root
    logger.propagate = False

    # 控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 文件Handler
    if not log_file:
        log_file = os.path.join(LOG_PATH, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)
    return logger


logger = get_logger()


def _take_over_uvicorn_loggers() -> None:
    """接管 uvicorn / hypercorn / watchfiles / py.warnings 等第三方 logger，统一格式与输出通道"""
    root = logging.getLogger()

    # 若 root 还没挂 handler，用 agent logger 的 handler 作为共用模板
    shared_handlers = logger.handlers
    if not shared_handlers:
        return

    shared_level = logging.DEBUG
    shared_formatter = DEFAULT_LOG_FORMAT

    third_party_loggers = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "hypercorn",
        "hypercorn.error",
        "hypercorn.access",
        "watchfiles",
        "py.warnings",
        "fastapi",
        "starlette",
    ]

    for name in third_party_loggers:
        lg = logging.getLogger(name)
        lg.setLevel(shared_level)
        # 清理自带 handler，避免与 root 重复打印
        for h in list(lg.handlers):
            lg.removeHandler(h)
        # 只 propagate 到 root（root 已挂共享 handler），自身不再挂 handler，避免一条日志打印两次
        lg.propagate = True

    # 捕获 warnings 到日志
    logging.captureWarnings(True)

    # 确保 root logger 也有 handler，兜底所有未配置的 logger
    if not root.handlers:
        root.setLevel(logging.DEBUG)
        for h in shared_handlers:
            root.addHandler(h)
            root.setLevel(max(root.level or logging.DEBUG, h.level or logging.DEBUG))
    else:
        # 已有 handler 的情况下也挂一份 formatter，保证格式统一
        for h in root.handlers:
            if not h.formatter:
                h.setFormatter(shared_formatter)


def _get_client_ip(request: Request) -> str:
    """尽量取真实客户端 IP（兼容反向代理 X-Forwarded-For）"""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("X-Real-IP")
    if xri:
        return xri
    host = getattr(request.client, "host", None) if request.client else None
    return host or "-"


async def _read_request_body(request: Request) -> bytes:
    """
    安全读取 request body。

    注意：必须用 request.body()，不能手动 stream()。
    FastAPI 的 @app.middleware("http") 实际注册的是 BaseHTTPMiddleware，
    它通过 _CachedRequest.wrapped_receive 给下游转发 body：
      - 中间件调用了 body()  → 下游能拿到完整 _body
      - 中间件调用了 stream() → _stream_consumed=True，下游只会拿到空 body（导致 422）
    所以这里用 request.body() 缓存，下游 FastAPI 才能再次读到 body。
    """
    state = request.scope.setdefault("state", {})
    if "body" in state:
        return state["body"]
    body = await request.body()
    state["body"] = body
    return body


def setup_fastapi_logging(
        app: FastAPI,
        *,
        log_request_line: bool = True,
        log_request_body: bool = True,
        log_response_body: bool = False,
        log_headers: bool = False,
        access_logger_name: str = "agent.access"
) -> None:
    """
    一键为 FastAPI 配置 API 调用日志（不需要再单独声明中间件）。
    包含：
      - HTTP 请求/响应日志中间件（method/path/query/body/status_code/耗时/client_ip）
      - 全局异常兜底处理器（未捕获异常也会打 error 日志并返回 500）
      - 接管 uvicorn/starlette/watchfiles/warnings 等第三方 logger，格式统一

    参数：
      - log_request_line: 是否打印「请求进入」日志（默认 True；设 False 可整行取消）
      - log_request_body:  请求日志里是否打印请求体（默认 True，自动跳过二进制）
      - log_response_body: 是否打印响应体（默认 False；SSE 流式接口请保持 False）
      - log_headers:       是否打印完整请求头（默认 False，headers 噪音大，还容易带 http2-settings 等乱码）

    用法（main.py 中）：
        from fastapi import FastAPI
        from app.utils.logger_handle import setup_fastapi_logging

        app = FastAPI(...)
        setup_fastapi_logging(app)
        app.include_router(...)
    """
    app_id = id(app)
    if app_id in _SETUP_APPS:
        logger.debug("setup_fastapi_logging: app 已配置过，跳过重复初始化")
        return

    # 1) 先接管第三方 logger
    _take_over_uvicorn_loggers()
    access_logger = get_logger(access_logger_name)

    # 2) 注册 HTTP 请求日志中间件（基于 Starlette BaseHTTPMiddleware 模式，手写 add_middleware）
    @app.middleware("http")
    async def _http_access_log_middleware(request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        method = request.method
        url = request.url.path
        query = request.url.query or ""
        client_ip = _get_client_ip(request)

        # 请求头（默认不打印，仅 log_headers=True 时输出并脱敏）
        safe_headers = None
        if log_headers:
            headers_dict = dict(request.headers)
            safe_headers = _mask_sensitive(headers_dict)

        req_body_text = ""
        if log_request_body and method in ("POST", "PUT", "PATCH", "DELETE"):
            # 只对有 Content-Type 暗示为文本/JSON 的请求读取 body；二进制上传跳过
            ct = request.headers.get("Content-Type", "")
            is_binary_stream = "octet-stream" in ct or "multipart" in ct
            if not is_binary_stream:
                try:
                    raw = await _read_request_body(request)
                    if raw:
                        decoded = raw.decode("utf-8", errors="replace")
                        parsed = _safe_json_loads(decoded)
                        if parsed is not None:
                            masked = _mask_sensitive(parsed) if isinstance(parsed, dict) else parsed
                            req_body_text = json.dumps(masked, ensure_ascii=False)
                        else:
                            req_body_text = decoded
                    else:
                        # body 为空：给出诊断提示，常见于 HTTP/2 h2c upgrade（uvicorn/httptools 不支持）
                        upgrade = request.headers.get("upgrade", "").lower()
                        cl = request.headers.get("content-length")
                        if upgrade == "h2c":
                            req_body_text = (
                                "<EMPTY> h2c(HTTP/2) upgrade 请求，uvicorn(httptools) 不支持 HTTP/2，"
                                "body 已丢失；请在客户端强制 HTTP/1.1 (Java: HttpClient.Version.HTTP_1_1)>"
                            )
                        elif cl and cl.isdigit() and int(cl) > 0:
                            req_body_text = f"<EMPTY> content-length={cl} 但未读到 body"
                except Exception as e:
                    req_body_text = f"<read body failed: {type(e).__name__}: {e}>"

        if log_request_line:
            req_log_parts = [
                f"{client_ip}",
                f"-> {method} {url}" + (f"?{query}" if query else ""),
            ]
            if safe_headers is not None:
                req_log_parts.append(f"headers={json.dumps(safe_headers, ensure_ascii=False)}")
            if req_body_text:
                req_log_parts.append(f"body={_truncate(req_body_text)}")
            access_logger.info(" | ".join(req_log_parts))

        try:
            response = await call_next(request)
        except Exception as exc:
            # 全局未捕获异常兜底（FastAPI 本身也会捕获，但这里保证 error 日志 + 统一响应体）
            duration_ms = (time.perf_counter() - start) * 1000
            tb = traceback.format_exc()
            access_logger.error(
                f"{client_ip} <- {method} {url} | status=500 | duration_ms={duration_ms:.2f} | "
                f"exc={type(exc).__name__}: {exc}\n{tb}"
            )
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal Server Error: {type(exc).__name__}"},
            )

        duration_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code

        resp_body_text = ""
        if log_response_body:
            # 读取 response body 并重新塞回去
            try:
                resp_raw = [chunk async for chunk in response.body_iterator]
                response.body_iterator = iterate_in_threadpool(iter(resp_raw))
                merged = b"".join(resp_raw)
                if merged:
                    decoded = merged.decode("utf-8", errors="replace")
                    parsed = _safe_json_loads(decoded)
                    if parsed is not None:
                        resp_body_text = json.dumps(parsed, ensure_ascii=False)
                    else:
                        resp_body_text = decoded
            except Exception:
                resp_body_text = ""

        resp_log_parts = [
            f"{client_ip}",
            f"<- {method} {url}",
            f"status={status_code}",
            f"duration_ms={duration_ms:.2f}",
        ]
        if resp_body_text:
            resp_log_parts.append(f"resp={_truncate(resp_body_text)}")

        # 4xx/5xx 用 warn/error 级别，其余 info
        if status_code >= 500:
            access_logger.error(" | ".join(resp_log_parts))
        elif status_code >= 400:
            access_logger.warning(" | ".join(resp_log_parts))
        else:
            access_logger.info(" | ".join(resp_log_parts))

        return response

    # 3) 全局异常处理器（兜底 HTTPException 之外的未捕获异常）
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception):
        tb = traceback.format_exc()
        logger.error(
            f"Unhandled exception at {request.method} {request.url.path}: "
            f"{type(exc).__name__}: {exc}\n{tb}"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {type(exc).__name__}"},
        )

    _SETUP_APPS.add(app_id)
    logger.info(
        f"setup_fastapi_logging done. log_request_body={log_request_body}, "
        f"log_response_body={log_response_body}"
    )
