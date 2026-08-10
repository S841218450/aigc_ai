import time
import traceback
from typing import Any, Callable, Awaitable
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ExtendedModelResponse
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage
from app.utils.logger_handle import logger


class LLMMonitorMiddleware(AgentMiddleware):
    def __init__(self, node_name: str, thread_id: str, user_id: str):
        self.node_name = node_name
        self.thread_id = thread_id
        self.user_id = user_id
        self.run_start_map = {}
        # 按 thread_id 聚合一次 agent 执行期间所有次 LLM 调用的 token
        self._token_accumulator: dict[str, dict[str, int]] = {}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse | AIMessage:
        # 修复：使用 request.id 替代不存在的 request.runtime.run_id
        execution_info = request.runtime.execution_info
        run_id = execution_info.run_id if execution_info else "unknown_run"
        start_ts = time.time()
        self.run_start_map[run_id] = start_ts

        # 1. 模型调用前日志
        messages = request.messages
        max_tokens_config = request.model_settings.get("max_tokens", 2048)

        logger.info(
            f"【Agent LLM发起调用】threadId={self.thread_id} userId={self.user_id} node={self.node_name}",
            extra={
                "run_id": run_id,
                "max_tokens_config": max_tokens_config
            }
        )

        try:
            response: ModelResponse | AIMessage = await handler(request)

            # 2. 调用成功，解析token用量
            cost_sec = round(time.time() - self.run_start_map.pop(run_id, time.time()), 3)

            # 从 AIMessage.usage_metadata 中提取 token 信息
            usage_metadata = None
            llm_raw_output = ""
            if isinstance(response, AIMessage):
                usage_metadata = response.usage_metadata
                llm_raw_output = response.content or ""
            elif isinstance(response, ExtendedModelResponse):
                inner = response.model_response
                if inner and inner.result:
                    for msg in inner.result:
                        if isinstance(msg, AIMessage):
                            usage_metadata = msg.usage_metadata
                            llm_raw_output = msg.content or ""
                            break
            else:  # ModelResponse
                if response.result:
                    for msg in response.result:
                        if isinstance(msg, AIMessage):
                            usage_metadata = msg.usage_metadata
                            llm_raw_output = msg.content or ""
                            break

            usage_metadata = usage_metadata or {}
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            output_token_details = usage_metadata.get("output_token_details", {}) or {}
            reasoning_tokens = output_token_details.get("reasoning", 0)
            total_tokens = usage_metadata.get("total_tokens", prompt_tokens + completion_tokens)

            # 累加到 thread_id 维度的汇总
            acc = self._token_accumulator.setdefault(self.thread_id, {
                "prompt_tokens": 0, "completion_tokens": 0,
                "reasoning_tokens": 0, "total_tokens": 0, "llm_calls": 0,
            })
            acc["prompt_tokens"] += prompt_tokens
            acc["completion_tokens"] += completion_tokens
            acc["reasoning_tokens"] += reasoning_tokens
            acc["total_tokens"] += total_tokens
            acc["llm_calls"] += 1

            logger.info(
                f"【Agent LLM调用完成】threadId={self.thread_id} userId={self.user_id} node={self.node_name} 耗时={cost_sec}s "
                f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
                f"reasoning_tokens={reasoning_tokens} total_tokens={total_tokens}",
                extra={
                    "run_id": run_id,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "output_content": llm_raw_output[:2000]
                }
            )

            # 推理Token超限预警
            if reasoning_tokens >= max_tokens_config * 0.8:
                logger.warning(
                    f"【LLM推理Token预警】threadId={self.thread_id} node={self.node_name} "
                    f"reasoning_tokens={reasoning_tokens} 接近上限 max_tokens={max_tokens_config}",
                    extra={"threadId": self.thread_id, "node": self.node_name, "run_id": run_id}
                )
            return response

        except Exception as err:
            # 3. 捕获异常日志
            cost_sec = round(time.time() - self.run_start_map.pop(run_id, time.time()), 3)
            err_stack = traceback.format_exc()

            logger.error(
                f"【Agent LLM调用异常】threadId={self.thread_id} userId={self.user_id} node={self.node_name} 耗时={cost_sec}s",
                extra={
                    "run_id": run_id,
                    "error_class": type(err).__name__,
                    "error_msg": str(err),
                    "error_stack": err_stack
                }
            )
            raise err

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # 每次 agent 执行前重置 token 汇总
        self._token_accumulator[self.thread_id] = {
            "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "total_tokens": 0, "llm_calls": 0,
        }
        return None

    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        acc = self._token_accumulator.pop(self.thread_id, None)
        if acc and acc["llm_calls"] > 0:
            logger.info(
                f"【Agent执行完成-TOKEN汇总】threadId={self.thread_id} userId={self.user_id} node={self.node_name} "
                f"LLM调用次数={acc['llm_calls']} prompt_tokens={acc['prompt_tokens']} "
                f"completion_tokens={acc['completion_tokens']} reasoning_tokens={acc['reasoning_tokens']} "
                f"total_tokens={acc['total_tokens']}",
                extra={
                    "threadId": self.thread_id,
                    "userId": self.user_id,
                    "node": self.node_name,
                    **acc,
                }
            )
        return None