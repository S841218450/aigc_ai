# ============================================================
# AIGC Platform - FastAPI 应用镜像
# ============================================================
FROM python:3.14-slim

# 国内 pip 镜像加速（海外服务器可删除这两行）
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# chromadb / onnxruntime 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依赖层：requirements.txt 未变化时复用 Docker 层缓存，避免每次全量重装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 应用层：源码变更只重新拷贝，不重装依赖
COPY app ./app

# 运行时数据目录（logs / chroma / sqlite，部署时建议挂载 volume 持久化）
RUN mkdir -p /app/logs /app/chroma_db /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
