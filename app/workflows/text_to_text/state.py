from typing import Optional, Dict, Any
from pydantic import BaseModel


class TextToTextState(BaseModel):
    prompt: str
    params: Dict[str, Any] = {}
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None