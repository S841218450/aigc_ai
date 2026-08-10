from typing import Any, Optional
from pydantic import BaseModel


class Result(BaseModel):
    code: int = 200
    msg: str = "操作成功"
    success: bool = True
    data: Optional[Any] = None

    @staticmethod
    def ok(data: Any = None, msg: str = "操作成功") -> dict:
        return Result(code=200, msg=msg, data=data).model_dump()

    @staticmethod
    def fail(code: int = 500, msg: str = "操作失败") -> dict:
        return Result(code=code, msg=msg, success=False).model_dump()
