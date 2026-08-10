from fastapi import APIRouter, Request
from app.models.schemas.common import Result
from app.models.schemas.text_to_image import TextToImageRequest, HumanTextToImageRequest, RetryRequest
from app.workflows.text_to_image.graph import TextToImageGraph
from app.services.sse_service import (
    SSEService, graph_to_sse_events, replay_events, build_sse_response, task_manager,
)
from app.services.node_status import TEXT_TO_IMAGE_NODE_MAP, build_node_data
from app.services.event_store import event_store

router = APIRouter()


def _build_node_data(node_name: str, state_update: dict) -> dict:
    return build_node_data(node_name, state_update, node_map=TEXT_TO_IMAGE_NODE_MAP)


async def _create_sse_response(threadId, stream, last_event_id=None):
    """通用 SSE 响应构建：断点续传 + 流式输出"""
    sse = SSEService(thread_id=threadId, event_store=event_store)

    async def event_generator():
        if last_event_id is not None:
            async for chunk in replay_events(str(threadId), last_event_id, sse, event_store):
                yield chunk
        async for chunk in sse.stream(graph_to_sse_events(
            stream, sse, threadId,
            node_map=TEXT_TO_IMAGE_NODE_MAP,
            build_node_data=_build_node_data,
        )):
            yield chunk

    return build_sse_response(event_generator())


@router.post("/generate")
async def generate_image(request: TextToImageRequest, req: Request):
    """流式生成图片 - SSE（支持断点续传）"""
    if not request.userId:
        return Result.fail(code=1001, msg="请登录")

    # 解析 Last-Event-ID 头（断点续传）
    last_event_id = req.headers.get("Last-Event-ID")
    last_seq_id = int(last_event_id) if last_event_id else None

    graph = TextToImageGraph()
    stream = graph.run_stream(request.prompt, request.userId, request.threadId, request.model, request.params)

    return await _create_sse_response(request.threadId, stream, last_seq_id)


@router.post("/select")
async def select_image(request: HumanTextToImageRequest, req: Request):
    """人工介入后流式恢复执行 - SSE（支持断点续传）"""
    if not request.userId or not request.threadId:
        return Result.fail(code=1001, msg="核心参数错误")

    last_event_id = req.headers.get("Last-Event-ID")
    last_seq_id = int(last_event_id) if last_event_id else None

    graph = TextToImageGraph()
    stream = graph.human_back_stream(request.user_select, request.userId, request.threadId)

    return await _create_sse_response(request.threadId, stream, last_seq_id)


@router.post("/retry")
async def retry_workflow(request: RetryRequest, req: Request):
    """出错后重试执行：从上次中断的节点继续 - SSE"""
    if not request.userId or not request.threadId:
        return Result.fail(code=1001, msg="核心参数错误")

    last_event_id = req.headers.get("Last-Event-ID")
    last_seq_id = int(last_event_id) if last_event_id else None

    graph = TextToImageGraph()
    stream = graph.retry_stream(request.userId, request.threadId)

    return await _create_sse_response(request.threadId, stream, last_seq_id)


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
