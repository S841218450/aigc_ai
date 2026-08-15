from langgraph.graph import StateGraph, END
from app.workflows.knowledge_base.state import KnowledgeBaseState
from app.workflows.knowledge_base.nodes import (
    intent_recognition_node,
    chat_answer_node,
    retrieval_agent_node,
    answer_node,
    confidence_evaluation_node,
    format_response_node,
)
from app.workflows.knowledge_base.router import kb_needed_router


class KnowledgeBaseGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(KnowledgeBaseState)

        workflow.add_node("intent_recognition", intent_recognition_node)  # 意图识别（路由节点：闲聊 vs 查库）
        workflow.add_node("chat_answer", chat_answer_node)  # 闲聊/功能咨询（无需检索，直接回答结束）
        workflow.add_node("retrieval_agent", retrieval_agent_node)  # 检索 agent（工具查证据，推进度）
        workflow.add_node("answer", answer_node)  # 回答节点（基于检索证据流式生成）
        workflow.add_node("confidence_evaluation", confidence_evaluation_node)  # 置信度评估
        workflow.add_node("format_response", format_response_node)  # 格式化响应（只补 sources）

        workflow.set_entry_point("intent_recognition")

        # 意图识别 → 条件路由：闲聊/功能咨询直接 chat_answer 结束（省一次完整 agent 调用）；
        # 其余（含结构化/枚举/统计类）全部进检索 agent，由 agent 自主决定查文档还是查表
        workflow.add_conditional_edges(
            "intent_recognition",
            kb_needed_router,
            {
                "chat_answer": "chat_answer",
                "retrieval_agent": "retrieval_agent",
            },
        )
        workflow.add_edge("chat_answer", END)

        # 检索 agent → 回答节点 → 置信度评估 → 格式化响应（检索与生成解耦，证据经 state 显式传递）
        workflow.add_edge("retrieval_agent", "answer")
        workflow.add_edge("answer", "confidence_evaluation")
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
            async for event in self.graph.astream(initial_state, stream_mode=["updates", "custom"]):
                # 多模式流返回 (mode, chunk)；节点内 StreamWriter 发来的 custom 信号（节点开始/流式增量）转成 node_start
                if isinstance(event, tuple):
                    mode, chunk = event
                else:
                    mode, chunk = "updates", event
                if mode == "custom":
                    yield "node_start", chunk
                    continue
                for node_name, state_update in chunk.items():
                    yield node_name, state_update
        except Exception as e:
            import traceback
            print(f"[知识库流异常] err={traceback.format_exc()}")
            yield "error", {"msg": str(e)}
