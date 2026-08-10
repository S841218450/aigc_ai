
from typing import Any, AsyncGenerator, Optional
import asyncio
import uuid
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.workflows.text_to_image.state import TextToImageState
from app.workflows.text_to_image.nodes import (
    generate_image_node,
    desc_code_judge_node,
    decision_router,
    supplementary_node,
    summer_node, prompt_combined_node, human_interrupt_node
)
from app.services.checkpointer import checkpointer_service
from app.core.middleware import WorkflowStatusMiddleware
from app.tools.image_generation.work_status import change_work_status
from app.utils.logger_handle import logger


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

    - selectList 选择题 → pending（等待用户补充描述，payload 带选择题）
    - summer_node（评估完即流程结束，已取消重绘回流）→ completed
    - 其他节点 → None：不刷屏 generating，保持当前状态即可
    """
    if not isinstance(state_update, dict):
        return None, None
    if state_update.get("selectList") not in (None, []):
        return "pending", state_update.get("selectList")
    if node_name == "summer_node":
        return "completed", None
    return None, None


class TextToImageGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(TextToImageState)

        # 1. 注册全部节点
        workflow.add_node("desc_code_judge_node", desc_code_judge_node) #描述判断节点
        workflow.add_node("decision_router", decision_router) #决策节点
        workflow.add_node("supplementary_node", supplementary_node) #补充描述节点
        workflow.add_node("summer_node", summer_node) #总结节点
        workflow.add_node("generate_image_node", generate_image_node) #生图节点
        workflow.add_node("human_interrupt_node", human_interrupt_node) #用户中断节点
        workflow.add_node("prompt_combined_node", prompt_combined_node) #提示词合并节点

        workflow.set_entry_point("desc_code_judge_node")
        workflow.add_edge("desc_code_judge_node", "decision_router")

        # 2. 决策节点条件分支
        def route_decision(state: TextToImageState):
            is_pass = state.get("isPass", False)
            if is_pass:
                return "generate_image_node"
            return "supplementary_node"

        workflow.add_conditional_edges("decision_router", route_decision)


        workflow.add_edge("supplementary_node", "human_interrupt_node")
        # 3. 补充描述后，合并提示词再进入生图节点
        def route_interrupt(state: TextToImageState):
            selectResult = state.get("selectResult", None)
            # P1#4 防御：记录 supplementary 被触发次数，超过 3 次自动放行（防止用户一直传 None）
            loop_count = state.get("supplementary_loop_count", 0)
            if selectResult is not None or loop_count >= 3:
                return "prompt_combined_node"
            return "supplementary_node"
        workflow.add_conditional_edges("human_interrupt_node", route_interrupt)


        workflow.add_edge("prompt_combined_node", "generate_image_node")
        # 4. 生图后判断：有图→进入总结评估，无图（生图失败/空返回）→直接结束
        def route_after_generate(state: TextToImageState):
            if state.get("image_url"):
                return "summer_node"
            return END

        workflow.add_conditional_edges("generate_image_node", route_after_generate)

        # 5. 总结节点：只评估图片质量并返回修改建议，不再回流重绘（已取消重绘机制）
        workflow.add_edge("summer_node", END)

        # 7. 编译图 (MongoDB 持久化 checkpointer)
        checkpointer = checkpointer_service.get_checkpointer()
        return workflow.compile(checkpointer=checkpointer)

    def _make_config(self, userId, threadId):
        return {"configurable": {
            "thread_id": threadId,
            "user_id": userId,
        }}

    def _make_initial_state(self, question, userId, threadId,model, params) -> TextToImageState:
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
            image_url=None,
            raw_image_urls=None,
            metadata=None,
            upload_retried=None,
            supplementary_loop_count=0,
            redraw_count=0,
            need_redraw=None,
            match_score=None,
            image_problem=None,
            modify_suggest=None,
            judge_note=None,
            answer=''
        )

    # ---- 流式（SSE 用） ----

    async def run_stream(self, question: str, userId: str, threadId: str = None, model: str = None, params: dict = None) -> AsyncGenerator:
        """流式执行：每完成一个节点 yield 一次 (node_name, state_snapshot)。

        业务状态统一在 graph 执行层显式推送（中间件不再散弹调用 change_work_status）：
        开始 → generating；pending/completed 由节点事件推断；正常结束 → completed；异常 → failed。
        """
        initial_state = self._make_initial_state(question, userId, threadId, model, params)
        threadId = initial_state["threadId"]
        config = self._make_config(initial_state.get("userId"), threadId)
        try:
            status_mw = WorkflowStatusMiddleware(threadId=threadId)
            await _safe_change_status(threadId, "generating")
            last_status = None
            async for event in status_mw.wrap_astream(
                self.graph.astream(initial_state, config, stream_mode=["updates", "custom"]),
            ):
                # 多模式流返回 (mode, chunk)；节点内 StreamWriter 发来的 custom 信号转成 node_start
                if isinstance(event, tuple):
                    mode, chunk = event
                else:
                    mode, chunk = "updates", event
                if mode == "custom":
                    yield "node_start", chunk
                    continue
                for node_name, state_update in chunk.items():
                    yield node_name, state_update
                    status, payload = _infer_work_status(node_name, state_update)
                    if status:
                        last_status = status
                        await _safe_change_status(threadId, status, payload)
            # 流正常走完：未出现 pending/failed 等终态才算真正完成
            if last_status in (None, "generating"):
                await _safe_change_status(threadId, "completed")
        except Exception as e:
            import traceback
            stack = traceback.format_exc()
            print(f"[启动流异常] threadId={threadId}, err={stack}")
            await _safe_change_status(threadId, "failed")
            yield "error", {"msg": str(e), "stack": stack}

    async def human_back_stream(self, user_select, userId, threadId) -> AsyncGenerator:
        """流式恢复执行 Command 标准方案（含 P1#4 supplementary 循环计数）"""
        config = self._make_config(userId, threadId)

        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            yield "error", {"msg": f"线程{threadId}不存在，请先发起生成流程"}
            return

        # P1#4：记录 supplementary 循环计数（human_interrupt 恢复时累加一次）
        current_values = dict(snapshot.values)
        prev_loop = current_values.get("supplementary_loop_count", 0) or 0
        update_state = {"supplementary_loop_count": prev_loop + 1}

        try:
            # 先把计数器写回 checkpoint，再 resume
            await self.graph.aupdate_state(config, update_state)
            status_mw = WorkflowStatusMiddleware(threadId=threadId)
            await _safe_change_status(threadId, "generating")
            last_status = None
            async for event in status_mw.wrap_astream(
                self.graph.astream(Command(resume=user_select), config, stream_mode=["updates", "custom"]),
            ):
                if isinstance(event, tuple):
                    mode, chunk = event
                else:
                    mode, chunk = "updates", event
                if mode == "custom":
                    yield "node_start", chunk
                    continue
                for node_name, state_update in chunk.items():
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
            print(f"[人工恢复流异常] threadId={threadId}, err={stack}")
            await _safe_change_status(threadId, "failed")
            yield "error", {"msg": str(e), "stack": stack}

    async def retry_stream(self, userId, threadId) -> AsyncGenerator:
        """出错后重试执行：从最后一次成功的 checkpoint 继续，重新执行出错节点 - SSE"""
        config = self._make_config(userId, threadId)

        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            yield "error", {"msg": f"线程{threadId}不存在，请先发起生成流程"}
            return

        try:
            status_mw = WorkflowStatusMiddleware(threadId=threadId)
            await _safe_change_status(threadId, "generating")
            last_status = None
            # astream(None) 从 checkpoint 继续，出错节点会重新执行
            async for event in status_mw.wrap_astream(
                self.graph.astream(None, config, stream_mode=["updates", "custom"]),
            ):
                if isinstance(event, tuple):
                    mode, chunk = event
                else:
                    mode, chunk = "updates", event
                if mode == "custom":
                    yield "node_start", chunk
                    continue
                for node_name, state_update in chunk.items():
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
