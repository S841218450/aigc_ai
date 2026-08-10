from langgraph.graph import StateGraph, END
from app.workflows.knowledge_base.state import KnowledgeBaseState
from app.workflows.knowledge_base.nodes import (
    intent_recognition_node,
    query_understanding_node,
    chat_answer_node,
    kb_needed_router,
    structured_query_node,
    structured_router,
    bm25_retrieval_node,
    vector_retrieval_node,
    hybrid_merge_node,
    rerank_node,
    context_building_node,
    generate_answer_node,
    confidence_evaluation_node,
    format_response_node,
)


class KnowledgeBaseGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(KnowledgeBaseState)

        workflow.add_node("intent_recognition", intent_recognition_node)  # 意图识别（路由节点，独立串行以保证 SRP）
        workflow.add_node("chat_answer", chat_answer_node)  # 闲聊/功能咨询（无需检索，直接回答结束）
        workflow.add_node("structured_query", structured_query_node)  # 结构化查表（枚举/计数/筛选/推荐，失败回退检索）
        workflow.add_node("query_understanding", query_understanding_node)  # 查询理解（重写+实体抽取合并）
        workflow.add_node("bm25_retrieval", bm25_retrieval_node)  # BM25检索
        workflow.add_node("vector_retrieval", vector_retrieval_node)  # 向量检索
        workflow.add_node("hybrid_merge", hybrid_merge_node)  # 混合合并（同时作为两路并行检索的 join 点）
        workflow.add_node("rerank", rerank_node)  # 重排序
        workflow.add_node("context_building", context_building_node)  # 上下文构建
        workflow.add_node("generate_answer", generate_answer_node)  # 生成回答
        workflow.add_node("confidence_evaluation", confidence_evaluation_node)  # 置信度评估
        workflow.add_node("format_response", format_response_node)  # 格式化响应

        workflow.set_entry_point("intent_recognition")

        # 第一阶段：意图识别 → 条件路由
        # 需要知识库检索 → 查询理解；闲聊/功能咨询（needs_retrieval=False）→ chat_answer 直接结束，
        # 避免每轮对话都触发检索造成资源消耗；结构化类问题（枚举/计数/筛选/推荐）→ 确定性查表
        workflow.add_conditional_edges(
            "intent_recognition",
            kb_needed_router,
            {
                "query_understanding": "query_understanding",
                "structured_query": "structured_query",
                "chat_answer": "chat_answer",
            },
        )
        workflow.add_edge("chat_answer", END)

        # 结构化查询分支：成功（结构化上下文就绪）→ 直接生成答案；失败 → 回退查询理解走向量检索
        workflow.add_conditional_edges(
            "structured_query",
            structured_router,
            {
                "generate_answer": "generate_answer",
                "query_understanding": "query_understanding",
            },
        )

        # 第二阶段：查询理解 → BM25 / 向量 并行扇出
        # LangGraph 对同一源节点的多条出边会自动并行执行
        workflow.add_edge("query_understanding", "bm25_retrieval")
        workflow.add_edge("query_understanding", "vector_retrieval")

        # 第三阶段：双路召回检索合并
        workflow.add_edge("bm25_retrieval", "hybrid_merge")
        workflow.add_edge("vector_retrieval", "hybrid_merge")

        # 第四阶段：精排 + 生成 + 校验 + 格式化（串行，链路有严格先后依赖）
        workflow.add_edge("hybrid_merge", "rerank")
        workflow.add_edge("rerank", "context_building")
        workflow.add_edge("context_building", "generate_answer")
        workflow.add_edge("generate_answer", "confidence_evaluation")
        workflow.add_edge("confidence_evaluation", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile(debug=True)

    async def run_stream(self, query: str, params: dict = None, thread_id: str = None, user_id: str = None,
                         kb_id: str = None, filter_folder_ids: list = None, filter_doc_ids: list = None,
                         chat_history: list = None, conversation_summary: str = None,
                         attachments: list = None, messages: list = None):
        initial_state = KnowledgeBaseState(
            question=query,
            params=params or {},
            threadId=thread_id or "",
            userId=user_id or "",
            kb_id=kb_id,
            filter_folder_ids=filter_folder_ids,
            filter_doc_ids=filter_doc_ids,
            chat_history=chat_history or [],
            conversation_summary=conversation_summary or "",
            attachments=attachments or None,
            messages=messages or [],
            answer=None,
            query_rewritten=None,
            intent_type=None,
            intent_confidence=None,
            needs_retrieval=None,
            bm25_docs=None,
            vector_docs=None,
            hybrid_docs=None,
            reranked_docs=None,
            documents=None,
            kg_entities=None,
            kg_validated=None,
            context=None,
            confidence_score=None,
            has_reliable_source=None,
            sources=None,
            retrieval_strategy=None,
            next_step=None,
            structured_context=None,
            structured_failed=None,
        )
        try:
            async for event in self.graph.astream(initial_state, stream_mode="updates"):
                for node_name, state_update in event.items():
                    yield node_name, state_update
        except Exception as e:
            import traceback
            print(f"[知识库流异常] err={traceback.format_exc()}")
            yield "error", {"msg": str(e)}
