import time
import json
import traceback
from typing import Any, Callable, Awaitable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.prebuilt.tool_node import ToolCallRequest
from app.utils.logger_handle import logger


# 工具返回值在日志中的最大长度，避免大 payload 污染日志
MAX_TOOL_OUTPUT_LOG_LEN = 2000
# 慢工具阈值（秒），超过会打 warning
SLOW_TOOL_THRESHOLD_SEC = 5.0
# 默认工具重试次数（仅对被标记的临时性错误生效）
DEFAULT_TOOL_RETRY = 1
# 被认定为"临时性错误"的异常类型，会触发重试
TRANSIENT_ERROR_CLASSES = (TimeoutError, ConnectionError,)


def _truncate_output(content: Any, max_len: int = MAX_TOOL_OUTPUT_LOG_LEN) -> str:
    """把工具返回值转成可打印字符串并截断，避免日志爆炸"""
    try:
        if isinstance(content, (dict, list, tuple)):
            text = json.dumps(content, ensure_ascii=False, default=str)
        else:
            text = str(content)
    except Exception:
        text = repr(content)
    if len(text) <= max_len:
        return text or "<EMPTY>"
    return text[:max_len] + f"...(truncated, total {len(text)} chars)"


class ToolMonitorMiddleware(AgentMiddleware):
    """工具调用可观测性中间件。

    功能：
    - 记录每次工具调用的名称/入参/耗时/返回/异常堆栈
    - 慢工具阈值预警
    - agent 执行结束后输出工具维度的汇总日志（调用次数/成功数/失败数/总耗时）
    - （可选）对网络/超时类临时性错误自动重试
    """

    def __init__(
        self,
        node_name: str,
        thread_id: str,
        user_id: str,
        *,
        slow_tool_threshold_sec: float = SLOW_TOOL_THRESHOLD_SEC,
        retry_on_transient: int = DEFAULT_TOOL_RETRY,
    ):
        self.node_name = node_name
        self.thread_id = thread_id
        self.user_id = user_id
        self.slow_tool_threshold_sec = slow_tool_threshold_sec
        self.retry_on_transient = max(0, int(retry_on_transient))
        # 汇总：按 thread_id -> {tool_name -> {calls, success, failed, total_cost_sec}}
        self._tool_summary: dict[str, dict[str, dict[str, Any]]] = {}

    # -------- 生命周期钩子 --------
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """每次 agent 执行前重置汇总"""
        self._tool_summary[self.thread_id] = {}
        return None

    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """agent 执行结束后输出工具调用汇总"""
        summary = self._tool_summary.pop(self.thread_id, None)
        if not summary:
            logger.info(
                f"【Agent执行完成-工具汇总】threadId={self.thread_id} userId={self.user_id} "
                f"node={self.node_name} 本次执行未调用任何工具"
            )
            return None

        total_calls = 0
        total_success = 0
        total_failed = 0
        total_cost = 0.0
        per_tool_parts = []
        for tool_name, stats in summary.items():
            total_calls += stats["calls"]
            total_success += stats["success"]
            total_failed += stats["failed"]
            total_cost += stats["total_cost_sec"]
            avg_cost = round(stats["total_cost_sec"] / max(1, stats["calls"]), 3)
            per_tool_parts.append(
                f"{tool_name}[{stats['calls']}次 ok={stats['success']} "
                f"fail={stats['failed']} avg={avg_cost}s total={round(stats['total_cost_sec'], 3)}s]"
            )

        logger.info(
            f"【Agent执行完成-工具汇总】threadId={self.thread_id} userId={self.user_id} node={self.node_name} "
            f"工具调用总次数={total_calls} 成功={total_success} 失败={total_failed} "
            f"总耗时={round(total_cost, 3)}s | 明细: {' ; '.join(per_tool_parts)}",
            extra={
                "threadId": self.thread_id,
                "userId": self.user_id,
                "node": self.node_name,
                "total_tool_calls": total_calls,
                "tool_success": total_success,
                "tool_failed": total_failed,
                "tool_total_cost_sec": round(total_cost, 3),
                "per_tool": summary,
            },
        )
        return None

    # -------- 工具调用拦截 --------
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        tool_name = tool_call.get("name") or "<UNKNOWN>"
        tool_args = tool_call.get("args") or {}
        tool_call_id = tool_call.get("id") or ""

        # 1. 工具调用前日志
        tool_instance = getattr(request, "tool", None)
        tool_description = (
            (getattr(tool_instance, "description", "") or "")[:200]
            if tool_instance is not None else ""
        )
        logger.info(
            f"【Agent工具调用开始】threadId={self.thread_id} userId={self.user_id} node={self.node_name} "
            f"tool={tool_name} call_id={tool_call_id} args={json.dumps(tool_args, ensure_ascii=False)}",
            extra={
                "threadId": self.thread_id,
                "userId": self.user_id,
                "node": self.node_name,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "tool_args": tool_args,
                "tool_description": tool_description,
            },
        )

        start_ts = time.time()
        attempts = 0
        last_err: Exception | None = None

        # 2. 执行工具（支持重试临时性错误）
        while attempts <= self.retry_on_transient:
            attempts += 1
            try:
                result = await handler(request)
                cost_sec = round(time.time() - start_ts, 3)
                output_preview = ""
                if isinstance(result, ToolMessage):
                    output_preview = _truncate_output(result.content)
                elif isinstance(result, Command):
                    output_preview = f"<Command:update keys={list(getattr(result, 'update', {}).keys()) or []}>"
                else:
                    output_preview = f"<Unknown: {type(result).__name__}>"

                # 慢工具预警
                if cost_sec >= self.slow_tool_threshold_sec:
                    logger.warning(
                        f"【Agent工具慢调用预警】threadId={self.thread_id} node={self.node_name} "
                        f"tool={tool_name} 耗时={cost_sec}s 超过阈值={self.slow_tool_threshold_sec}s",
                        extra={
                            "tool_name": tool_name, "cost_sec": cost_sec,
                            "threshold": self.slow_tool_threshold_sec,
                        },
                    )

                logger.info(
                    f"【Agent工具调用完成】threadId={self.thread_id} userId={self.user_id} node={self.node_name} "
                    f"tool={tool_name} call_id={tool_call_id} 耗时={cost_sec}s "
                    f"result={output_preview}",
                    extra={
                        "threadId": self.thread_id,
                        "userId": self.user_id,
                        "node": self.node_name,
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "cost_sec": cost_sec,
                        "tool_output_preview": output_preview,
                    },
                )

                # 写入汇总
                self._record_summary(tool_name, cost_sec, success=True)
                return result

            except TRANSIENT_ERROR_CLASSES as transient_err:
                last_err = transient_err
                remaining = self.retry_on_transient - (attempts - 1)
                if remaining <= 0:
                    break
                cost_so_far = round(time.time() - start_ts, 3)
                logger.warning(
                    f"【Agent工具调用重试】threadId={self.thread_id} node={self.node_name} "
                    f"tool={tool_name} 第{attempts}次失败(类型={type(transient_err).__name__})，"
                    f"已耗时{cost_so_far}s，剩余重试次数={remaining}",
                    extra={"tool_name": tool_name, "attempt": attempts},
                )
                continue

            except Exception as err:
                # 非临时性错误，直接走异常分支
                last_err = err
                break

        # 3. 工具执行失败（重试耗尽或非临时异常）
        cost_sec = round(time.time() - start_ts, 3)
        err_stack = traceback.format_exc()
        err_class = type(last_err).__name__ if last_err else "UnknownError"
        err_msg = str(last_err) if last_err else ""

        logger.error(
            f"【Agent工具调用异常】threadId={self.thread_id} userId={self.user_id} node={self.node_name} "
            f"tool={tool_name} call_id={tool_call_id} 耗时={cost_sec}s 重试次数={attempts - 1} "
            f"error={err_class}: {err_msg}",
            extra={
                "threadId": self.thread_id,
                "userId": self.user_id,
                "node": self.node_name,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "cost_sec": cost_sec,
                "retry_count": attempts - 1,
                "error_class": err_class,
                "error_msg": err_msg,
                "error_stack": err_stack,
            },
        )

        # 失败记录也写入汇总，随后抛出异常让上层 handle_tool_errors / agent 处理
        self._record_summary(tool_name, cost_sec, success=False)
        raise last_err if last_err is not None else RuntimeError(err_msg or "Tool execution failed")

    # -------- 内部工具方法 --------
    def _record_summary(self, tool_name: str, cost_sec: float, *, success: bool) -> None:
        bucket = self._tool_summary.setdefault(self.thread_id, {})
        stats = bucket.setdefault(tool_name, {
            "calls": 0, "success": 0, "failed": 0, "total_cost_sec": 0.0,
        })
        stats["calls"] += 1
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        stats["total_cost_sec"] += cost_sec
