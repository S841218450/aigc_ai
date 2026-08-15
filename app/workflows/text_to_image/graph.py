
from typing import Any, AsyncGenerator, Optional
import uuid
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.core.middleware import WorkflowStatusMiddleware
from app.services.checkpointer import checkpointer_service
from app.workflows.common.retry import MAX_MANUAL_RETRIES, retry_node, with_auto_retry
from app.workflows.common.status import infer_work_status, safe_change_status
from app.workflows.text_to_image.nodes import (
    decision_node,
    generate_node,
    input_check_node,
    interrupt_node,
    prompt_optimize_node,
    supplementary_node,
)
from app.workflows.text_to_image.state import TextToImageState


class TextToImageGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(TextToImageState)

        # 注册全部节点（统一包一层自动重试：节点抛异常先自动重试，仍失败进入手动重试；
        # 中断类节点 interrupt_node / retry_node 不能包，interrupt() 需要透传给 LangGraph 运行时）
        workflow.add_node("input_check_node", with_auto_retry(input_check_node))  # 输入检查
        workflow.add_node("decision_node", with_auto_retry(decision_node))  # 方案决策
        workflow.add_node("supplementary_node", with_auto_retry(supplementary_node))  # 补充描述
        workflow.add_node("interrupt_node", interrupt_node)  # 补充描述中断门
        workflow.add_node("prompt_optimize_node", with_auto_retry(prompt_optimize_node))  # 提示词优化
        workflow.add_node("generate_node", with_auto_retry(generate_node))  # 生图
        workflow.add_node("retry_node", retry_node)  # 手动重试中断门

        # 每个节点后接条件路由：有 node_error 就进重试中断门，否则走下一个节点
        def make_route(next_node: str):
            def route(state: TextToImageState):
                if state.get("node_error"):
                    return "retry_node"
                return next_node
            return route

        # 决策路由：描述不通过则进入补充描述流程（文生图特有的提示词补充决策流程）
        def route_decision(state: TextToImageState):
            if state.get("node_error"):
                return "retry_node"
            if state.get("isPass", False):
                return "generate_node"
            return "supplementary_node"

        # 补充中断恢复路由：用户已选择或循环超限则放行，否则重新生成选择题
        def route_interrupt(state: TextToImageState):
            selectResult = state.get("selectResult", None)
            # P1#4 防御：记录 supplementary 被触发次数，超过 3 次自动放行（防止用户一直传 None）
            loop_count = state.get("supplementary_loop_count", 0)
            if selectResult is not None or loop_count >= 3:
                return "prompt_optimize_node"
            return "supplementary_node"

        # 手动重试路由：回到失败的节点；无目标或超过重试上限则终止（防死循环）
        def route_retry(state: TextToImageState):
            retry_target = state.get("retry_target")
            retry_count = state.get("retry_count", 0) or 0
            if not retry_target or retry_count >= MAX_MANUAL_RETRIES:
                return END
            return retry_target

        workflow.set_entry_point("input_check_node")
        workflow.add_conditional_edges("input_check_node", make_route("decision_node"))
        workflow.add_conditional_edges("decision_node", route_decision)
        workflow.add_conditional_edges("supplementary_node", make_route("interrupt_node"))
        workflow.add_conditional_edges("interrupt_node", route_interrupt)
        workflow.add_conditional_edges("prompt_optimize_node", make_route("generate_node"))
        workflow.add_conditional_edges("generate_node", make_route(END))
        workflow.add_conditional_edges("retry_node", route_retry)

        # 编译图 (MongoDB 持久化 checkpointer，手动重试依赖它恢复中断)
        checkpointer = checkpointer_service.get_checkpointer()
        return workflow.compile(checkpointer=checkpointer)

    def _make_config(self, userId, threadId):
        return {"configurable": {
            "thread_id": threadId,
            "user_id": userId,
        }}

    def _make_initial_state(self, question, userId, threadId, model, params) -> TextToImageState:
        """初始化初始状态"""
        return TextToImageState(
            question=question.strip(),
            userId=userId,
            threadId=threadId or uuid.uuid7(),
            prompt='',
            model=model,
            params=params or {},
            messages=[],
            agent_log=None,
            totalScope=None,
            need_manual_count=None,
            judgeList=None,
            judge_summary=None,
            selectResult=None,
            isPass=None,
            decide_result=None,
            selectList=None,
            image_list=None,
            metadata=None,
            supplementary_loop_count=0,
            answer='',
            node_error=None,
            retry_target=None,
            retry_count=0,
        )

    # ---- 流式（SSE 用） ----

    async def run_stream(self, question: str, userId: str, threadId: str = None, model: str = None, params: dict = None) -> AsyncGenerator:
        """流式执行：每完成一个节点 yield 一次 (node_name, state_snapshot)。

        业务状态统一在 graph 执行层显式推送（中间件不再散弹调用 change_work_status）：
        开始 → generating；节点失败/中断 → failed；generate_node（最后节点）→ completed；异常 → failed。
        """
        initial_state = self._make_initial_state(question, userId, threadId, model, params)
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

    async def human_back_stream(self, user_select, userId, threadId) -> AsyncGenerator:
        """流式恢复执行 Command 标准方案（含 P1#4 supplementary 循环计数）"""
        config = self._make_config(userId, threadId)

        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            yield "error", {"failed_node": None}
            return

        # P1#4：记录 supplementary 循环计数（interrupt_node 恢复时累加一次）
        current_values = dict(snapshot.values)
        prev_loop = current_values.get("supplementary_loop_count", 0) or 0
        update_state = {"supplementary_loop_count": prev_loop + 1}

        try:
            # 先把计数器写回 checkpoint，再 resume
            await self.graph.aupdate_state(config, update_state)
            status_mw = WorkflowStatusMiddleware(threadId=threadId)
            await safe_change_status(threadId, "generating")
            last_status = None
            last_node = None
            async for event in status_mw.wrap_astream(
                self.graph.astream(Command(resume=user_select), config, stream_mode=["updates", "custom"]),
            ):
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
            print(f"[人工恢复流异常] threadId={threadId}, err={stack}")
            await safe_change_status(threadId, "failed")
            yield "error", {"failed_node": last_node}

    async def retry_stream(self, userId, threadId) -> AsyncGenerator:
        """手动重试：用 Command(resume=True) 唤醒 retry_node 的中断，回到失败节点继续执行 - SSE"""
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
