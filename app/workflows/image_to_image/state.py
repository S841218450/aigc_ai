from typing import Dict, Any, List, Optional

from pydantic import BaseModel

from app.workflows.common.imageState import ImageStatus


class ImageItemType(BaseModel):
    id: str = ""
    url: str = ""


class ImageToImageState(ImageStatus):
    # ---------------------- 图生图输入字段 ----------------------
    originImageList: List[ImageItemType]  # 参考图列表

    # ---------------------- params_filter_node 参数过滤输出 ----------------------
    clean_prompt: Optional[str]      # 剔除敏感词/图像参数词后的提示词
    filter_reason: Optional[str]     # 过滤说明

    # ---------------------- generate_image_node 生图节点输出 ----------------------
    image_list: Optional[List[str]]      # 最终图片地址列表
    metadata: Optional[Dict[str, Any]]   # 生图元数据

    # ---------------------- quality_evaluation_node 质量评估输出 ----------------------
    isPass: Optional[bool]        # 质量是否合格
    match_score: Optional[int]    # 与提示词匹配度 0-10
    image_problem: Optional[str]  # 图片存在的缺陷

    # ---------------------- 节点重试机制 ----------------------
    node_error: Optional[str]     # 最近一次节点执行错误信息（有值说明需要重试）
    retry_target: Optional[str]   # 需要重试的节点名
    retry_count: Optional[int]    # 手动重试轮数（防死循环）
