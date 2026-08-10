"""工作流状态观测中间件。
本模块两个中间件只保留观测职责（日志），不再触碰业务状态接口：
1. WorkflowStatusMiddleware（Graph 级）: 包装 graph.astream，记录节点完成与异常日志
2. AgentStatusMiddleware（Agent 级）: 记录 agent 生命周期、LLM/工具调用次数与耗时
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Awaitable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.prebuilt.tool_node import ToolCallRequest

from app.utils.logger_handle import logger


# ================================================================  Graph 层


class WorkflowStatusMiddleware:
    """LangGraph 级别的观测中间件（只记录日志，不推送业务状态）。

    用法：在 graph.py 的 run_stream / human_back_stream 里：

        mw = WorkflowStatusMiddleware(threadId=threadId)
        async for event in mw.wrap_astream(self.graph.astream(...)):
            yield event

    业务状态（generating/completed/failed/pending）由 graph.py 在流循环里
    按顺序显式推送，不再由中间件代为调用。
    """

    def __init__(self, threadId: str):
        self.threadId = threadId
        # 每个 threadId 维度的耗时统计
        self._node_start_ts: dict[str, float] = {}

    # ---- 对外：把 astream 包起来 ----

    async def wrap_astream(self, astream_coro):
        """包裹 `graph.astream(..., stream_mode="updates")`，记录节点完成 / 异常日志。"""
        try:
            async for event in astream_coro:
                if not isinstance(event, dict):
                    yield event
                    continue
                for node_name, state_update in event.items():
                    self._log_node_done(node_name, state_update)
                yield event
        except Exception as exc:
            stack = traceback.format_exc()
            logger.error(
                f"[GraphStatus] stream exception threadId={self.threadId}: {exc!r}",
                extra={"error_stack": stack},
            )
            raise

    # ---- 内部：日志 ----

    def _log_node_done(self, node_name: str, state_update: Any) -> None:
        agent_log = ""
        if isinstance(state_update, dict):
            agent_log = (state_update.get("agent_log") or "")[:200]
        logger.info(
            f"【Graph节点完成】threadId={self.threadId} node={node_name} agent_log={agent_log}",
            extra={
                "threadId": self.threadId,
                "node": node_name,
                "agent_log": agent_log,
            },
        )


# ==============================================================  Agent 层


class AgentStatusMiddleware(AgentMiddleware):
    """`create_agent(middleware=[...])` 里使用的 Agent 级观测中间件（只记录日志）。

    和 LLMMonitorMiddleware / ToolMonitorMiddleware 是并列的兄弟：
    - abefore_agent / aafter_agent: 生命周期钩子
    - awrap_model_call: 统计 LLM 调用次数
    - awrap_tool_call: 统计工具调用次数与耗时

    注意：它不再推送业务状态。graph 层面的业务状态由各 workflow 的 graph.py
    在 run_stream / human_back_stream 里显式维护。
    """

    def __init__(
        self,
        node_name: str,
        thread_id: str,
        user_id: str,
    ):
        self.node_name = node_name
        self.thread_id = thread_id
        self.user_id = user_id
        self._model_call_count = 0
        self._tool_call_count = 0

    # ---- Agent 生命周期 ----

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        logger.info(
            f"【AgentStatus-启动】threadId={self.thread_id} userId={self.user_id} node={self.node_name}",
            extra={"threadId": self.thread_id, "user_id": self.user_id, "node": self.node_name},
        )
        return None

    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        logger.info(
            f"【AgentStatus-结束】threadId={self.thread_id} node={self.node_name} "
            f"model_calls={self._model_call_count} tool_calls={self._tool_call_count}",
            extra={
                "threadId": self.thread_id,
                "node": self.node_name,
                "model_calls": self._model_call_count,
                "tool_calls": self._tool_call_count,
            },
        )
        return None

    # ---- LLM 调用统计 ----

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse | AIMessage]],
    ) -> ModelResponse | AIMessage:
        self._model_call_count += 1
        try:
            return await handler(request)
        except Exception:
            raise

    # ---- 工具调用统计 ----

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        self._tool_call_count += 1
        start = time.time()
        try:
            return await handler(request)
        except Exception:
            raise
        finally:
            cost = round(time.time() - start, 3)
            logger.info(
                f"【AgentStatus-工具耗时】threadId={self.thread_id} node={self.node_name} "
                f"tool={request.tool_call.get('name')} 耗时={cost}s",
                extra={"threadId": self.thread_id, "node": self.node_name, "cost_sec": cost},
            )
