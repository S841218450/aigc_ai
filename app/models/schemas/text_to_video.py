from typing import Optional, Dict, Any
from pydantic import BaseModel


class TextToVideoRequest(BaseModel):
    prompt: str
    params: Optional[Dict[str, Any]] = {}


class TextToVideoResponse(BaseModel):
    video_url: str
    metadata: Optional[Dict[str, Any]] = None