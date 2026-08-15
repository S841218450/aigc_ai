# -*- coding: utf-8 -*-
"""知识库工作流路由：节点间的分支选择（LangGraph conditional edges）。"""
from app.workflows.knowledge_base.state import KnowledgeBaseState


def kb_needed_router(state: KnowledgeBaseState) -> str:
    """意图识别后路由：不需要知识库检索（闲聊/功能咨询）直接 chat_answer 结束；
    其余（含结构化/枚举/统计类）全部进主查询 agent，由 agent 自主决定查文档还是查表。"""
    # 缺省/未识别时保守走闲聊分支（chat_answer），避免误触发知识库检索
    if not state.get("needs_retrieval", False):
        return "chat_answer"
    return "retrieval_agent"
