from typing import Optional, Dict, Any
from pydantic import BaseModel


class TextToVideoState(BaseModel):
    prompt: str
    params: Dict[str, Any] = {}
    video_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None