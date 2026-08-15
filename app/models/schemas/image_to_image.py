from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OriginImageItem(BaseModel):
    id: str = ""
    url: str = ""


class ImageToImageRequest(BaseModel):
    userId: str = Field(default=None, description="用户id")
    threadId: str = Field(default=None, description="线程id")
    type: str = Field(default="image", description="工作类型")
    prompt: str = Field(description="用户原始提示词")
    model: Optional[str] = Field(default="default", description="生图模型（如 qwen-image-3.0-pro / default）")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="图像参数：imageSize/imageQty/referenceIntensity")
    originImageList: Optional[List[OriginImageItem]] = Field(default_factory=list, description="参考图列表（图生图输入）")



class ImageToImageRetryRequest(BaseModel):
    threadId: str = Field(default=None, description="线程id")
    userId: str = Field(default=None, description="用户id")
