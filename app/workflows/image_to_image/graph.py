import uuid
from typing import Any, AsyncGenerator, List, Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.core.middleware import WorkflowStatusMiddleware
from app.services.checkpointer import checkpointer_service
from app.tools.image_generation.work_status import change_work_status
from app.utils.logger_handle import logger
from app.workflows.image_to_image.nodes import (
    MAX_MANUAL_RETRIES,
    await_retry_node,
    generate_image_node,
    params_filter_node,
    prompt_optimization_node,
    quality_evaluation_node,
    summary_node,
    with_auto_retry,
)
from app.workflows.image_to_image.state import ImageToImageState


async def _safe_change_status(threadId: str, status: str, data: Any = None) -> None:
    """graph 执行层按序推送业务状态：await 保证顺序，避免 fire-and-forget 乱序改写终态。

    change_work_status 内部已吞掉 HTTP 异常，这里是兜底；失败只记日志，不影响主流程。
    """
    try:
        await change_work_status(threadId, status, data)
    except Exception as e:
        logger.warning("状态推送失败 threadId=%s status=%s err=%s", threadId, status, e)


def _infer_work_status(node_name: str, state_update: Any) -> tuple[Optional[str], Any]:
    """节点事件 → 业务状态推断，返回 (status, payload)；status 为 None 表示保持当前状态。

    - 节点失败中断（__interrupt__ 带 retry_target）→ failed（等待手动重试）
    - 节点自动重试耗尽（state_update 带 node_error/retry_target）→ failed
    - summary_node（图生图流程最终节点）→ completed
    - selectList 选择题 → pending（通用规则，图生图当前未用）
    - 其他节点 → None：不刷屏 generating，保持当前状态即可
    """
    if node_name == "__interrupt__":
        value = state_update[0].value if state_update else {}
        if isinstance(value, dict) and value.get("retry_target"):
            return "failed", None
        return None, None
    if not isinstance(state_update, dict):
        return None, None
    if state_update.get("selectList") not in (None, []):
        return "pending", state_update.get("selectList")
    if state_update.get("node_error") or state_update.get("retry_target"):
        return "failed", None
    if node_name == "await_retry_node":
        return "failed", None
    if node_name == "summary_node":
        return "completed", None
    return None, None


class ImageToImageGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(ImageToImageState)

        # 注册全部节点（统一包一层自动重试：节点抛异常先自动重试，仍失败进入手动重试）
        workflow.add_node("params_filter_node", with_auto_retry(params_filter_node))  # 参数过滤
        workflow.add_node("prompt_optimization_node", with_auto_retry(prompt_optimization_node))  # 提示词优化
        workflow.add_node("generate_image_node", with_auto_retry(generate_image_node))  # 生图
        workflow.add_node("quality_evaluation_node", with_auto_retry(quality_evaluation_node))  # 质量评估
        workflow.add_node("summary_node", with_auto_retry(summary_node))  # 总结
        workflow.add_node("await_retry_node", await_retry_node)  # 手动重试中断门

        # 每个节点后接条件路由：有 node_error 就进重试中断门，否则走下一个节点
        def make_route(next_node: str):
            def route(state: ImageToImageState):
                if state.get("node_error"):
                    return "await_retry_node"
                return next_node
            return route

        workflow.set_entry_point("params_filter_node")
        workflow.add_conditional_edges("params_filter_node", make_route("prompt_optimization_node"))
        workflow.add_conditional_edges("prompt_optimization_node", make_route("generate_image_node"))
        workflow.add_conditional_edges("generate_image_node", make_route("quality_evaluation_node"))
        workflow.add_conditional_edges("quality_evaluation_node", make_route("summary_node"))
        workflow.add_conditional_edges("summary_node", make_route(END))

        # 手动重试路由：回到失败的节点；无目标或超过重试上限则终止（防死循环）
        def route_retry(state: ImageToImageState):
            retry_target = state.get("retry_target")
            retry_count = state.get("retry_count", 0) or 0
            if not retry_target or retry_count >= MAX_MANUAL_RETRIES:
                return END
            return retry_target

        workflow.add_conditional_edges("await_retry_node", route_retry)

        # 编译图 (MongoDB 持久化 checkpointer，手动重试依赖它恢复中断)
        checkpointer = checkpointer_service.get_checkpointer()
        return workflow.compile(checkpointer=checkpointer)

    def _make_config(self, userId: str, threadId: str) -> dict:
        return {"configurable": {"thread_id": threadId, "user_id": userId}}

    def _make_initial_state(
        self,
        prompt: str,
        userId: str,
        threadId: Optional[str],
        params: Optional[dict],
        originImageList: Optional[List[dict]],
    ) -> ImageToImageState:
        """初始化初始状态"""
        return ImageToImageState(
            question=prompt.strip(),
            userId=userId,
            threadId=threadId or str(uuid.uuid7()),
            prompt="",
            model="default",
            params=params or {},
            messages=[],
            answer="",
            originImageList=[
                {"id": str(item.get("id") or ""), "url": str(item.get("url") or "")}
                for item in (originImageList or [])
            ],
            agent_log=None,
            clean_prompt=None,
            filter_reason=None,
            image_list=None,
            metadata=None,
            isPass=None,
            match_score=None,
            image_problem=None,
            node_error=None,
            retry_target=None,
            retry_count=0,
        )

    # ---- 流式（SSE 用） ----

    async def run_stream(
        self,
        prompt: str,
        userId: str,
        threadId: Optional[str] = None,
        params: Optional[dict] = None,
        originImageList: Optional[List[dict]] = None,
    ) -> AsyncGenerator:
        """流式执行：每完成一个节点 yield 一次 (node_name, state_snapshot)。

        业务状态统一在 graph 执行层显式推送（中间件不再散弹调用 change_work_status）：
        开始 → generating；节点失败/中断 → failed；summary_node 或正常结束 → completed；异常 → failed。
        """
        initial_state = self._make_initial_state(prompt, userId, threadId, params, originImageList)
        threadId = initial_state["threadId"]
        config = self._make_config(initial_state.get("userId"), threadId)
        try:
            status_mw = WorkflowStatusMiddleware(threadId=threadId)
            await _safe_change_status(threadId, "generating")
            last_status = None
            async for event in status_mw.wrap_astream(
                self.graph.astream(initial_state, config, stream_mode="updates"),
            ):
                for node_name, state_update in event.items():
                    yield node_name, state_update
                    status, payload = _infer_work_status(node_name, state_update)
                    if status:
                        last_status = status
                        await _safe_change_status(threadId, status, payload)
            # 流正常走完：未出现 failed/pending 等终态才算真正完成
            if last_status in (None, "generating"):
                await _safe_change_status(threadId, "completed")
        except Exception as e:
            import traceback
            stack = traceback.format_exc()
            print(f"[启动流异常] threadId={threadId}, err={stack}")
            await _safe_change_status(threadId, "failed")
            yield "error", {"msg": str(e), "stack": stack}

    async def retry_stream(self, userId: str, threadId: str) -> AsyncGenerator:
        """手动重试：用 Command(resume=True) 唤醒 await_retry_node 的中断，回到失败节点继续执行"""
        config = self._make_config(userId, threadId)

        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            yield "error", {"msg": f"线程{threadId}不存在，请先发起生成流程"}
            return

        try:
            status_mw = WorkflowStatusMiddleware(threadId=threadId)
            await _safe_change_status(threadId, "generating")
            last_status = None
            async for event in status_mw.wrap_astream(
                self.graph.astream(Command(resume=True), config, stream_mode="updates"),
            ):
                for node_name, state_update in event.items():
                    yield node_name, state_update
                    status, payload = _infer_work_status(node_name, state_update)
                    if status:
                        last_status = status
                        await _safe_change_status(threadId, status, payload)
            if last_status in (None, "generating"):
                await _safe_change_status(threadId, "completed")
        except Exception as e:
            import traceback
            stack = traceback.format_exc()
            print(f"[重试流异常] threadId={threadId}, err={stack}")
            await _safe_change_status(threadId, "failed")
            yield "error", {"msg": str(e), "stack": stack}
