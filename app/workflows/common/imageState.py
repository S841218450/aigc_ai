from typing import Optional

from app.workflows.common.baseState import BaseState
from app.models.schemas.text_to_image import paramsType


class ImageStatus(BaseState):
    # ---------------------- 用户输入基础字段 ----------------------
    prompt: str  # 最终用于生图的优化提示词
    params: paramsType  # 图像生成参数（尺寸/张数/参考强度等）
    model: str  # 模型


    # ---------------------- Agent 执行日志（每节点追加） ----------------------
    agent_log: Optional[str]  # 节点执行摘要，SSE 回传给前端
