from typing import Literal, Optional
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()


def _strip_secret(value: Optional[str]) -> Optional[str]:
    """去掉环境变量值首尾的引号。

    部分部署环境（如非 dotenv 的 .env 解析/脚本注入）不会剥掉 .env 里的引号，
    会把 QWEN_API_KEY='"sk-xxx"' 原样带进环境变量，导致请求携带 Bearer "sk-xxx"
    被 DashScope 拒绝（401 invalid_api_key）。这里统一兜底。
    """
    if not value:
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


# 统一环境常量抽离
QWEN_API_KEY = _strip_secret(os.getenv("QWEN_API_KEY"))
QWEN_BASE_URL = os.getenv("QWEN_API_BASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CLOUDBASE_ACCESS_TOKEN = os.getenv("CLOUDBASE_ACCESS_TOKEN")
ARK_API_KEY = os.getenv("ARK_API_KEY")

MAX_RETRIES = 3

# 补充缺失的 Specialist 字面量
ModelType = Literal[
    'summarizer',
    'Executor',
    'embedding',
    'intent',
    'inspiration',
    'Supervisor',
    'generate_image',
    'supplementary',
    'prompt_combined',
    'Specialist',
    'Reviewer',
    'text_to_image',
    'kb_answer',
]


def get_image_client() -> AsyncOpenAI:
    """获取火山引擎图片生成专用异步客户端"""
    return AsyncOpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
    )

def get_model(role: ModelType, config: Optional[dict] = None):
    # 通用大模型基础配置，减少重复代码
    base_chat_cfg = {
        "api_key": QWEN_API_KEY,
        "base_url": QWEN_BASE_URL,
        "max_retries": MAX_RETRIES
    }

    if role == "Supervisor":
        # ReAct调度中枢，大幅提升输出上限，支持多轮思考+多工具调用
        return ChatOpenAI(
            model="qwen3.7-max-2026-05-17",
            temperature=0.2,
            max_tokens=8192,
            streaming=True,
            **base_chat_cfg
        )
    elif role == "Specialist":
        # 任务拆解，中等输出上限
        return ChatOpenAI(
            model="qwen3.7-max-2026-06-08",
            temperature=0.3,
            max_tokens=2048,
            **base_chat_cfg
        )
    elif role == "kb_answer":
        # 知识库回答节点：长回答输出（检索与回答已拆分，回答单独占用输出预算）
        return ChatOpenAI(
            model="qwen3.7-max-2026-06-08",
            temperature=0.3,
            max_tokens=4096,
            **base_chat_cfg
        )
    elif role == "Reviewer":
        # 结果校验
        return ChatOpenAI(
            model="glm-5.2",
            temperature=0.2,
            max_tokens=2048,
            **base_chat_cfg
        )
    elif role == "intent":
        # 结构化抽取，4096足够
        return ChatOpenAI(
            model="qwen3.7-flash-2026-07-15",
            temperature=0.2,
            max_tokens=4096,
            **base_chat_cfg
        )
    elif role == "inspiration":
        # 每日灵感：高温度保证题材多样性，输出灵感种子
        return ChatOpenAI(
            model="qwen3.7-max-preview",
            temperature=0.9,
            max_tokens=2048,
            **base_chat_cfg
        )
    elif role == "supplementary":
        # 选择题生成，需要多道题+选项，输出较长
        return ChatOpenAI(
            model="qwen3.7-flash-2026-07-15",
            temperature=0.2,
            max_tokens=8192,
            **base_chat_cfg
        )
    elif role == "embedding":
        return OpenAIEmbeddings(
            model="qwen3.7-text-embedding",
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            chunk_size=20,
            check_embedding_ctx_length=False,
        )
    elif role in ("summarizer", "prompt_combined"):
        # 摘要、提示词合并，4096不变；config 可覆盖默认参数（如 temperature）
        kwargs = {
            "model": "qwen3.7-max-preview",
            "temperature": 0.5 if role == "summarizer" else 0.2,
            "max_tokens": 4096,
            **base_chat_cfg,
            **(config or {}),
        }
        return ChatOpenAI(**kwargs)

    else:
        # 兜底通用执行节点，从512上调至1024
        return ChatOpenAI(
            model="qwen3.7-max-preview",
            temperature=0.5,
            max_tokens=1024,
            **base_chat_cfg
        )