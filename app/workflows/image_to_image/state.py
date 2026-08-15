from typing import Dict, Any, List, Optional

from pydantic import BaseModel

from app.workflows.common.imageState import ImageStatus


class ImageItemType(BaseModel):
    id: str = ""
    url: str = ""


class ImageToImageState(ImageStatus):
    # ---------------------- 图生图输入字段 ----------------------
    originImageList: List[ImageItemType]  # 参考图列表

    # ---------------------- input_check_node 输入检查输出 ----------------------
    clean_prompt: Optional[str]      # 剔除敏感词/图像参数词后的提示词
    filter_reason: Optional[str]     # 过滤说明

    # ---------------------- generate_node 生图节点输出 ----------------------
    image_list: Optional[List[str]]      # 最终图片地址列表（统一数组，支持单张/多张）
    metadata: Optional[Dict[str, Any]]   # 生图元数据

    # ---------------------- 节点重试机制 ----------------------
    node_error: Optional[str]     # 最近一次节点执行错误信息（有值说明需要重试）
    retry_target: Optional[str]   # 需要重试的节点名
    retry_count: Optional[int]    # 手动重试轮数（防死循环）
