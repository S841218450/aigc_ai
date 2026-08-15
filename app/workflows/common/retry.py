"""节点自动/手动重试机制（文生图、图生图共用）。

- with_auto_retry：节点抛异常先自动重试（指数退避），仍失败则写入 node_error/retry_target，
  由图路由到 retry_node 等待用户手动重试（/retry 端点用 Command(resume=True) 恢复）。
- retry_node：手动重试中断门，暂停等待用户点击「重试」后回到失败节点继续执行。
"""
import asyncio
import functools
import inspect
import logging

from langgraph.types import StreamWriter, interrupt

from app.workflows.common.common_node import clean_return

logger = logging.getLogger(__name__)

MAX_AUTO_RETRIES = 2     # 节点自动重试次数（指数退避）
MAX_MANUAL_RETRIES = 3   # 手动重试总轮数上限（防死循环）


def with_auto_retry(node_fn, max_retries: int = MAX_AUTO_RETRIES, base_delay: float = 1.0):
    """节点自动重试装饰器：节点抛异常时自动重试，仍失败则写入 node_error/retry_target，
    由图路由到 retry_node 等待用户手动重试。
    """
    # 节点若声明了 writer 参数（SSE 节点开始信号），装饰器透传 LangGraph 注入的 writer
    node_accepts_writer = "writer" in inspect.signature(node_fn).parameters

    @functools.wraps(node_fn)
    async def wrapper(state, writer: StreamWriter = None):
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if node_accepts_writer:
                    result = await node_fn(state, writer=writer)
                else:
                    result = await node_fn(state)
                if result is None:
                    result = {}
                # 成功后清除上次失败标记，让路由走正常后续节点
                result["node_error"] = None
                result["retry_target"] = None
                return result
            except Exception as e:
                last_error = e
                logger.warning("[%s] 第 %s 次执行失败: %s", node_fn.__name__, attempt + 1, e)
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** attempt))
        return clean_return({
            "agent_log": f"{node_fn.__name__} 执行失败（自动重试 {max_retries} 次仍未成功），等待手动重试",
            "node_error": str(last_error),
            "retry_target": node_fn.__name__,
        })

    return wrapper


async def retry_node(state):
    """手动重试中断门：节点失败后暂停，等用户触发重试后回到失败节点继续执行。"""
    retry_count = (state.get("retry_count") or 0) + 1
    interrupt({
        "title": "步骤执行失败，等待重试",
        "message": "当前步骤执行失败，点击「重试」后将重新执行该步骤",
        "retry_target": state.get("retry_target"),
        "retry_count": retry_count,
    })
    return clean_return({
        "agent_log": f"用户触发第 {retry_count} 轮手动重试，正在重新执行失败步骤",
        "retry_count": retry_count,
    })
