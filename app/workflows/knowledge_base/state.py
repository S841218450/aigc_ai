from typing import Optional, Dict, Any, List

from app.workflows.common.baseState import BaseState


class KnowledgeBaseState(BaseState):
    params: Dict[str, Any]

    # ---------- 节点执行摘要（SSE 回传前端；未声明时 LangGraph 会从节点更新中丢弃该 key，
    # 导致 build_node_data 的 messages 分支回退到完整 answer，最后一条事件重复透传全文）----------
    agent_log: Optional[str]

    # ---------- 多轮对话记忆（滑动窗口现取 + 近期摘要，由 /query 组装后注入）----------
    chat_history: Optional[List[Dict[str, Any]]] = None
    # 超出滑动窗口的早期历史摘要（Agent 自持，增量滚动生成）
    conversation_summary: Optional[str] = None

    # ---------- 本次提问携带的附件（image 走多模态 / document 解析后注入上下文）----------
    attachments: Optional[List[Dict[str, Any]]] = None

    # ---------- 检索范围过滤（Java 显式传入 → 检索节点直接读取 ----------
    kb_id: Optional[str]
    filter_folder_ids: Optional[List[int]]
    filter_doc_ids: Optional[List[str]]

    query_rewritten: Optional[str]

    intent_type: Optional[str]
    intent_confidence: Optional[float]
    # 是否需要知识库检索（false 时走闲聊/功能咨询分支，跳过检索直接回答结束）
    needs_retrieval: Optional[bool]

    bm25_docs: Optional[List[Dict[str, Any]]]
    vector_docs: Optional[List[Dict[str, Any]]]
    hybrid_docs: Optional[List[Dict[str, Any]]]
    reranked_docs: Optional[List[Dict[str, Any]]]

    documents: Optional[List[Dict[str, Any]]]

    kg_entities: Optional[List[Dict[str, Any]]]
    kg_validated: Optional[bool]

    context: Optional[str]
    answer: Optional[str]

    confidence_score: Optional[float]
    has_reliable_source: Optional[bool]

    sources: Optional[List[Dict[str, Any]]]

    retrieval_strategy: Optional[str]
    next_step: Optional[str]

    # ---------- 结构化查询（枚举/计数/筛选/推荐类问题确定性查表）----------
    # 结果文本（直接作为 context 进入答案生成）；True 表示结构化查询失败需回退向量检索
    structured_context: Optional[str]
    structured_failed: Optional[bool]
