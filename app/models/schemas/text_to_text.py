from typing import Optional, Dict, Any
from pydantic import BaseModel


class TextToTextRequest(BaseModel):
    prompt: str
    params: Optional[Dict[str, Any]] = {}


class TextToTextResponse(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None