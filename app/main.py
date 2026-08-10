from fastapi import FastAPI
from app.api.v1 import router as api_v1_router
from app.utils.logger_handle import setup_fastapi_logging

app = FastAPI(
    title="AIGC Platform",
    description="AI Generated Content Platform with LangGraph and LangChain",
    version="1.0.0"
)

# 一键接入 API 调用日志（中间件 + 全局异常 + uvicorn 日志接管）
setup_fastapi_logging(app, log_request_body=True, log_response_body=False)

app.include_router(api_v1_router, prefix="/ai-api/v1")


@app.get("/")
async def root():
    return {"message": "AIGC Platform API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}