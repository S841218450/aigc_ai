from typing import Callable


from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.prebuilt.tool_node import ToolCallRequest
from app.utils.logger_handle import logger


@wrap_tool_call
def agent_tool(
        request: ToolCallRequest,
        handler:Callable[[ToolCallRequest],ToolMessage|Command]
)->ToolMessage | Command:
    logger.info(f"执行工具：{request.tool_call['name']}")
    logger.info(f"传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"工具：{request.tool_call['name']}调用成功")
        return result
    except Exception as e:
        logger.info(f"工具：{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model(
        state:AgentState,
        runtime:Runtime,
):
    logger.info(f"即将调用模型：共{len(state['messages'])}条消息")
    if state['messages']:
        logger.debug(f"{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None

