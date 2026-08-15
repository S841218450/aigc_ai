"""工作流业务状态推送（文生图、图生图共用）。

graph 执行层按序推送业务状态：开始 → generating；节点失败/中断 → failed；
流程最后节点 → completed；选择题 → pending；异常 → failed。
"""
from typing import Any, Optional

from app.tools.image_generation.work_status import change_work_status
from app.utils.logger_handle import logger


async def safe_change_status(threadId: str, status: str, data: Any = None) -> None:
    """graph 执行层按序推送业务状态：await 保证顺序，避免 fire-and-forget 乱序改写终态。

    change_work_status 内部已吞掉 HTTP 异常，这里是兜底；失败只记日志，不影响主流程。
    """
    try:
        await change_work_status(threadId, status, data)
    except Exception as e:
        logger.warning("状态推送失败 threadId=%s status=%s err=%s", threadId, status, e)


def infer_work_status(
    node_name: str,
    state_update: Any,
    final_node: str = "generate_node",
) -> tuple[Optional[str], Any]:
    """节点事件 → 业务状态推断，返回 (status, payload)；status 为 None 表示保持当前状态。

    - __interrupt__ 带 retry_target → failed（等待手动重试）
    - 节点自动重试耗尽（state_update 带 node_error/retry_target）→ failed
    - selectList 选择题 → pending（payload 带选择题）
    - retry_node（重试中断门）→ failed
    - final_node（各流程的最后一个节点，由调用方传入）→ completed
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
    if node_name == "retry_node":
        return "failed", None
    if node_name == final_node:
        return "completed", None
    return None, None
