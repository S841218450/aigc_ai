from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class TextToImageRequest(BaseModel):
    prompt: str
    userId: str
    threadId: str
    model: str = "default"
    params: Optional[Dict[str, Any]] = {}


class HumanTextToImageRequest(BaseModel):
    user_select: List[Dict[str, Any]]
    userId: str
    threadId: str

class RetryRequest(BaseModel):
    userId: str
    threadId: str

class paramsType(BaseModel):
    imageCount: int = Field(description="图片数量")
    imageProportion: str = Field(description="图片尺寸宽高比 16:9")
    imageQuality: str = Field(description="图片质量1080p/2k/4k")
    style: str = Field(description="图片风格")