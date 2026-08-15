from fastapi import APIRouter, Request

from app.models.schemas.common import Result
from app.models.schemas.image_to_image import ImageToImageRequest, ImageToImageRetryRequest
from app.services.event_store import event_store
from app.services.node_status import NODE_MAP, build_node_data
from app.services.sse_service import (
    SSEService,
    build_sse_response,
    graph_to_sse_events,
    replay_events,
    task_manager,
)
from app.workflows.image_to_image.graph import ImageToImageGraph

router = APIRouter()


async def _create_sse_response(threadId, stream, last_event_id=None):
    """通用 SSE 响应构建：断点续传 + 流式输出"""
    sse = SSEService(thread_id=threadId, event_store=event_store)

    async def event_generator():
        if last_event_id is not None:
            async for chunk in replay_events(str(threadId), last_event_id, sse, event_store):
                yield chunk
        async for chunk in sse.stream(graph_to_sse_events(
            stream, sse, threadId,
            node_map=NODE_MAP,
            build_node_data=build_node_data,
        )):
            yield chunk

    return build_sse_response(event_generator())


@router.post("/generate")
async def generate_image(request: ImageToImageRequest, req: Request):
    """图生图 - SSE 流式生成（支持断点续传）"""
    threadId = request.threadId
    if not threadId:
        return Result.fail(code=1001, msg="缺少工作id")

    # 解析 Last-Event-ID 头（断点续传）
    last_event_id = req.headers.get("Last-Event-ID")
    last_seq_id = int(last_event_id) if last_event_id else None

    graph = ImageToImageGraph()
    stream = graph.run_stream(
        prompt=request.prompt,
        userId=request.userId or "",
        threadId=threadId,
        params=request.params or {},
        originImageList=[item.model_dump() for item in (request.originImageList or [])],
        model=request.model or "default",
    )

    return await _create_sse_response(threadId, stream, last_seq_id)


@router.post("/retry")
async def retry_workflow(request: ImageToImageRetryRequest, req: Request):
    """重试执行：节点失败后，从失败节点继续执行 - SSE"""
    threadId = request.threadId
    if not threadId:
        return Result.fail(code=1001, msg="缺少工作id")

    last_event_id = req.headers.get("Last-Event-ID")
    last_seq_id = int(last_event_id) if last_event_id else None

    graph = ImageToImageGraph()
    stream = graph.retry_stream(request.userId or "", threadId)

    return await _create_sse_response(threadId, stream, last_seq_id)


@router.post("/cancel")
async def cancel_workflow(request: Request):
    """取消正在进行的工作流"""
    body = await request.json()
    threadId = body.get("threadId")
    if not threadId:
        return Result.fail(code=1001, msg="缺少 threadId")

    if task_manager.cancel(threadId):
        return Result.ok(msg="已发送取消指令")
    return Result.fail(code=404, msg="未找到进行中的任务")
