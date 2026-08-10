from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv
import os
load_dotenv()
class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    stability_api_key: Optional[str] = None
    
    # LLM Provider Settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    
    # Qwen Settings
    qwen_api_key: Optional[str] = None
    qwen_api_base_url: Optional[str] = None
    
    # Ollama Settings
    ollama_base_url: Optional[str] = None
    ollama_api_key: Optional[str] = None
    ollama_model: Optional[str] = None
    
    # LangSmith Settings
    langsmith_tracing: bool = False
    langsmith_endpoint: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
    
    # Database
    database_url: str = "sqlite:///./aigc_platform.db"

    # MongoDB (LangGraph Checkpointer)
    mongodb_url: str = os.getenv("MONGODB_URL")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME")

    # Java Backend API
    java_api_base_url: str = os.getenv("JAVA_API_BASE_URL")
    java_internal_token: str = os.getenv("JAVA_INTERNAL_TOKEN")
    
    # Vector Store
    vector_store_type: str = "chroma"
    chroma_persist_directory: str = "./chroma_db"
    
    # Embedding Model
    embedding_model: str = "text-embedding-ada-002"
    
    # Application
    debug: bool = False
    api_prefix: str = "/api/v1"

    # 腾讯云混元文生图
    cloudbase_api_key: Optional[str] = os.getenv("CLOUDBASE_API_KEY")
    cloudbase_env_id: Optional[str] = os.getenv("CLOUDBASE_ENV_ID")

    # 其他配置
    # COS 防盗链默认拒绝空 Referer，下载 COS 文件时带上该 Referer（匹配腾讯云防盗链白名单）
    # 通过 .env 的 COS_DOWNLOAD_REFERER 覆盖；留空则不带 Referer
    cos_download_referer: Optional[str] = os.getenv("COS_DOWNLOAD_REFERER") or "http://localhost"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow",  # Allow extra fields from .env
    )


settings = Settings()