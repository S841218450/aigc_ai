from typing import Optional, Dict, Any, Annotated, List, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel
from langchain_core.messages import BaseMessage


class BaseState(TypedDict):
    question: str #用户问题

    threadId: str #线程id
    userId: str #用户id

    messages: Annotated[List[BaseMessage], add_messages] #ai上下文
    answer: str #最终回答