from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class KnowledgeBaseSource(BaseModel):
    doc_name: str
    update_time: Optional[str] = ""
    section: Optional[str] = ""
    permission: Optional[str] = ""
    page: Optional[str] = ""
    doc_id: Optional[str] = ""
    score: Optional[float] = 0.0


class KnowledgeBaseChatHistoryItem(BaseModel):
    """Java 会话消息记录（原生格式，一条记录 = 一问一答；兼容精简 role/content 格式）
    说明：answer 为空代表该轮尚未回答（即当前进行中的消息，其 question 已由 /query 的 query 字段单独传入），
    Agent 解析时会整条跳过，避免与 query 重复注入。
    """
    id: Optional[str] = None
    sessionId: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    attachments: Optional[Any] = None
    status: Optional[int] = None
    errorMsg: Optional[str] = None
    createTime: Optional[str] = None
    # 兼容精简格式（role/content）
    role: Optional[str] = None
    content: Optional[str] = None


class KnowledgeBaseAttachment(BaseModel):
    type: str = Field(..., description="附件类型: image=图片（走多模态给模型看图） / document=文档（解析后注入上下文）")
    name: str = Field("", description="文件名（含后缀），用于类型推断和展示")
    url: Optional[str] = Field(None, description="可直接访问的附件 URL（图片/文档），Agent 优先下载使用")
    content: Optional[str] = Field(None, description="文本内容（小文档直接传文本兜底；图片可传 data URI 作为 image_url）")


class KnowledgeBaseQueryRequest(BaseModel):
    question: str
    threadId: Optional[str] = None
    userId: Optional[str] = None
    chat_history: Optional[List[KnowledgeBaseChatHistoryItem]] = None # 滑动窗口
    attachments: Optional[List[KnowledgeBaseAttachment]] = None  # 本次提问携带的附件（图片/文档）
    params: Optional[Dict[str, Any]] = {} # 其他参数
    # ---------- 以下为检索范围过滤字段（Java 端可直接传，不必塞 params）----------
    # 指定知识库（默认 default）。未传时从 params.kb_id 兜底
    kb_id: Optional[str] = None
    filter_folder_ids: Optional[List[int]] = None # 按目录过滤
    filter_doc_ids: Optional[List[str]] = None#按文档白名单过滤


class KnowledgeBaseQueryResponse(BaseModel):
    answer: str
    sources: Optional[List[KnowledgeBaseSource]] = None
    confidence_score: Optional[float] = None
    has_reliable_source: Optional[bool] = None
    intent_type: Optional[str] = None
    retrieval_strategy: Optional[str] = None


class StopGenerationRequest(BaseModel):
    """停止生成请求"""
    threadId: str = Field(..., description="要停止的会话/任务 ID（与 /query 传入的 threadId 一致）")


class RetryRequest(BaseModel):
    """重试请求：重新执行该会话最后一次用户查询"""
    threadId: str = Field(..., description="要重试的会话 ID")


class ControlResponse(BaseModel):
    success: bool
    threadId: str = ""
    message: str = ""


# ======================================================================
# 文档上传 / 入库 (Java 端回调 Agent 内部接口使用)
# ======================================================================

class KnowledgeBaseUploadRequest(BaseModel):
    """
    上传文档入库请求（兼容两种模式）：
      A. file_url 模式（推荐）：Java 传 URL 给 Agent，Agent 自己下载解析
      B. content 纯文本模式（兜底）：Java 先解析成文本再传（小 TXT/MD 用）
    """
    doc_id: str = Field(..., description="Java 端生成的唯一文档 ID，作为向量库/元数据的主键")
    user_id: str = Field(None, description="上传者/归属用户 ID（用户数据隔离维度，检索时按 userId 强过滤），必传")
    doc_name: str = Field(..., description="原始文件名（含后缀），用于展示和类型推断兜底")
    file_url: Optional[str] = Field(None, description="可直接访问的文件 URL（PDF/TXT/MD，和 doc_name 后缀一致），优先走此模式")
    content: Optional[str] = Field(None, description="兜底纯文本内容，file_url 缺失时使用")
    file_type: Optional[str] = Field("", description="类型兜底（pdf/txt/md），URL 和 doc_name 都没后缀时使用")
    file_md5: Optional[str] = Field("", description="Java 端预计算的文件 MD5，可选，用于幂等/去重校验")
    file_size: Optional[int] = Field(0, description="文件大小（字节），可选")
    kb_id: Optional[str] = Field("default", description="知识库 ID，预留多知识库场景，默认 default")
    
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Java 端透传的扩展元数据（权限、所属人、标签等）")
    force_reingest: Optional[bool] = Field(False, description="若 doc_id 已存在且状态 ready，是否强制重新分割入库（默认跳过）")


class KnowledgeBaseUploadResponse(BaseModel):
    success: bool
    doc_id: str
    chunk_count: int = 0
    skipped: bool = False
    skip_reason: str = ""
    file_type: str = ""
    file_md5: str = ""
    file_size: int = 0
    status: str = "processing"  # processing / ready / failed / skipped
    message: Optional[str] = ""


# ======================================================================
# 删除 / 批量删除
# ======================================================================

class DeleteResponse(BaseModel):
    success: bool = False
    doc_id: str = ""
    deleted_chunk_count: int = 0
    deleted_metadata: bool = False
    message: str = ""


class BatchDeleteRequest(BaseModel):
    doc_ids: List[str] = Field(..., description="要删除的 doc_id 列表，建议单次 <= 100")
    mode: Optional[str] = Field("hard", description="hard: 立即物理删（推荐）；soft: 标记删除留对账窗口")


class BatchDeleteItemResult(BaseModel):
    doc_id: str
    deleted_chunk_count: int = 0
    deleted_metadata: bool = False
    success: bool
    reason: str = ""


class BatchDeleteResponse(BaseModel):
    success: bool
    mode: str = "hard"
    total: int = 0
    success_count: int = 0
    failed_count: int = 0
    deleted_chunk_count: int = 0
    details: List[BatchDeleteItemResult] = Field(default_factory=list)
    message: str = ""


# ======================================================================
# 文档列表 / 详情
# ======================================================================

class DocumentListItem(BaseModel):
    doc_id: str
    kb_id: str = "default"
    doc_name: str = ""
    file_type: str = ""
    file_url: str = ""
    file_size: int = 0
    file_md5: str = ""
    chunk_count: int = 0
    status: str = ""  # processing/ready/failed/deleted
    fail_reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class DocumentListResponse(BaseModel):
    items: List[DocumentListItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50
