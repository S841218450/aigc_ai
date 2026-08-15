import uuid
from typing import Any, AsyncGenerator, List, Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.core.middleware import WorkflowStatusMiddleware
from app.services.checkpointer import checkpointer_service
from app.workflows.common.retry import MAX_MANUAL_RETRIES, retry_node, with_auto_retry
from app.workflows.common.status import infer_work_status, safe_change_status
from app.workflows.image_to_image.nodes import (
    generate_node,
    input_check_node,
    prompt_optimize_node,
)
from app.workflows.image_to_image.state import ImageToImageState


class ImageToImageGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(ImageToImageState)

        # 注册全部节点（统一包一层自动重试：节点抛异常先自动重试，仍失败进入手动重试；
        # 中断类节点 retry_node 不能包，interrupt() 需要透传给 LangGraph 运行时）
        workflow.add_node("input_check_node", with_auto_retry(input_check_node))  # 输入检查
        workflow.add_node("prompt_optimize_node", with_auto_retry(prompt_optimize_node))  # 提示词优化
        workflow.add_node("generate_node", with_auto_retry(generate_node))  # 生图
        workflow.add_node("retry_node", retry_node)  # 手动重试中断门

        # 每个节点后接条件路由：有 node_error 就进重试中断门，否则走下一个节点
        def make_route(next_node: str):
            def route(state: ImageToImageState):
                if state.get("node_error"):
                    return "retry_node"
                return next_node
            return route

        workflow.set_entry_point("input_check_node")
        workflow.add_conditional_edges("input_check_node", make_route("prompt_optimize_node"))
        workflow.add_conditional_edges("prompt_optimize_node", make_route("generate_node"))
        # 生成图片后直接结束
        workflow.add_conditional_edges("generate_node", make_route(END))

        # 手动重试路由：回到失败的节点；无目标或超过重试上限则终止（防死循环）
        def route_retry(state: ImageToImageState):
            retry_target = state.get("retry_target")
            retry_count = state.get("retry_count", 0) or 0
            if not retry_target or retry_count >= MAX_MANUAL_RETRIES:
                return END
            return retry_target

        workflow.add_conditional_edges("retry_node", route_retry)

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
        model: Optional[str] = None,
    ) -> ImageToImageState:
        """初始化初始状态"""
        return ImageToImageState(
            question=prompt.strip(),
            userId=userId,
            threadId=threadId or str(uuid.uuid7()),
            prompt="",
            model=model or "default",
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
        model: Optional[str] = None,
    ) -> AsyncGenerator:
        """流式执行：每完成一个节点 yield 一次 (node_name, state_snapshot)。

        业务状态统一在 graph 执行层显式推送（中间件不再散弹调用 change_work_status）：
        开始 → generating；节点失败/中断 → failed；generate_node（最后节点）→ completed；异常 → failed。
        """
        initial_state = self._make_initial_state(prompt, userId, threadId, params, originImageList, model)
        threadId = initial_state["threadId"]
        config = self._make_config(initial_state.get("userId"), threadId)
        try:
            status_mw = WorkflowStatusMiddleware(threadId=threadId)
            await safe_change_status(threadId, "generating")
            last_status = None
            last_node = None
            async for event in status_mw.wrap_astream(
                self.graph.astream(initial_state, config, stream_mode=["updates", "custom"]),
            ):
                # 多模式流返回 (mode, chunk)；节点内 StreamWriter 发来的 custom 信号转成 node_start
                if isinstance(event, tuple):
                    mode, chunk = event
                else:
                    mode, chunk = "updates", event
                if mode == "custom":
                    payload = chunk if isinstance(chunk, dict) else {}
                    last_node = payload.get("node") or last_node
                    yield "node_start", chunk
                    continue
                for node_name, state_update in chunk.items():
                    last_node = node_name
                    yield node_name, state_update
                    status, payload = infer_work_status(node_name, state_update, final_node="generate_node")
                    if status:
                        last_status = status
                        await safe_change_status(threadId, status, payload)
            # 流正常走完：未出现 failed/pending 等终态才算真正完成
            if last_status in (None, "generating"):
                await safe_change_status(threadId, "completed")
        except Exception as e:
            import traceback
            stack = traceback.format_exc()
            print(f"[启动流异常] threadId={threadId}, err={stack}")
            await safe_change_status(threadId, "failed")
            yield "error", {"failed_node": last_node}

    async def retry_stream(self, userId: str, threadId: str) -> AsyncGenerator:
        """手动重试：用 Command(resume=True) 唤醒 retry_node 的中断，回到失败节点继续执行"""
        config = self._make_config(userId, threadId)

        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            yield "error", {"failed_node": None}
            return

        try:
            status_mw = WorkflowStatusMiddleware(threadId=threadId)
            await safe_change_status(threadId, "generating")
            last_status = None
            last_node = None
            async for event in status_mw.wrap_astream(
                self.graph.astream(Command(resume=True), config, stream_mode=["updates", "custom"]),
            ):
                # 多模式流返回 (mode, chunk)；节点内 StreamWriter 发来的 custom 信号转成 node_start
                if isinstance(event, tuple):
                    mode, chunk = event
                else:
                    mode, chunk = "updates", event
                if mode == "custom":
                    payload = chunk if isinstance(chunk, dict) else {}
                    last_node = payload.get("node") or last_node
                    yield "node_start", chunk
                    continue
                for node_name, state_update in chunk.items():
                    last_node = node_name
                    yield node_name, state_update
                    status, payload = infer_work_status(node_name, state_update, final_node="generate_node")
                    if status:
                        last_status = status
                        await safe_change_status(threadId, status, payload)
            if last_status in (None, "generating"):
                await safe_change_status(threadId, "completed")
        except Exception as e:
            import traceback
            stack = traceback.format_exc()
            print(f"[重试流异常] threadId={threadId}, err={stack}")
            await safe_change_status(threadId, "failed")
            yield "error", {"failed_node": last_node}
