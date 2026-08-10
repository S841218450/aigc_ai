# ============================================================
# AIGC Platform - FastAPI 应用镜像
# ============================================================
FROM python:3.14-slim

WORKDIR /app

# chromadb / onnxruntime 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 只复制依赖清单与包源码；.env 不进入镜像（运行时通过环境变量注入，避免密钥泄漏）
COPY pyproject.toml ./
COPY app ./app

# 使用 pyproject.toml 完整依赖安装（requirements.txt 不完整，缺 httpx / pymongo 等）
RUN pip install --no-cache-dir .

# 运行时数据目录（logs / chroma / sqlite，部署时建议挂载 volume 持久化）
RUN mkdir -p /app/logs /app/chroma_db /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
