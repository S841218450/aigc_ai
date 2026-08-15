# -*- coding: utf-8 -*-
"""知识库工作流节点。

分层约定（设计新工作流时同样遵循）：
- graph.py：工作流接线（节点/路由组装）
- router.py：路由选择（条件分支）
- nodes.py：节点实现（本文件）
- tools/：工具调用，按用途区分文件（doc_tools.py 文档检索 / table_tools.py 表查询）
- app/workflows/common/：公共节点/工具（如 agent_stream.py 流式辅助、common_node.py agent 工厂）
"""
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.types import StreamWriter

from app.core.agents.model_factory import get_model
from app.core.prompts.prompts_factory import get_prompt
from app.tools.knowledge_tools.search_tools import query_kb_documents, query_kb_tables
from app.tools.knowledge_tools.time_tools import get_current_time

from app.workflows.common.agent_stream import agent_stream_answer
from app.workflows.common.common_node import create_workflow_agent
from app.workflows.knowledge_base.state import KnowledgeBaseState
from app.workflows.knowledge_base.tools.doc_tools import (
    build_doc_search_tool,
    build_list_documents_tool,
    kb_scope,
)
from app.workflows.knowledge_base.tools.table_tools import build_table_tools

logger = logging.getLogger(__name__)


class IntentRecognitionOutput(BaseModel):
    intent_type: str = Field(description="意图类型")
    confidence: float = Field(description="置信度 0-1")
    analysis: str = Field(description="分析说明")
    needs_retrieval: bool = Field(description="是否检索知识库：仅当问题需要知识库新增内容且历史无答案时为 true；闲聊/问候/系统功能咨询/历史上下文可直接回答的追问一律为 false。必须显式填写，不得省略")


# 意图识别节点：根据用户查询识别意图类型，并判断是否需要知识库检索
async def intent_recognition_node(state: KnowledgeBaseState, writer: StreamWriter) -> KnowledgeBaseState:
    llm = get_model('intent')
    writer({"node": "intent_recognition_node", "messages": "正在识别用户问题"})
    query = state.get("question", "")
    # 对话历史经消息轮次传入（与 create_agent 一致），不文本拼接进 prompt：
    # 系统提示词 + 历史消息（含摘要/多轮 user-assistant）+ 当前问题
    messages: List[BaseMessage] = [
        SystemMessage(content=get_prompt('knowledge', 'intent_recognition')),
        *(state.get("messages") or []),
        HumanMessage(content=query),
    ]

    structured_llm = llm.with_structured_output(IntentRecognitionOutput)
    result: IntentRecognitionOutput = await structured_llm.ainvoke(messages)

    # 模型漏填字段时保守默认 False（宁走闲聊分支，也不误触发知识库检索）
    needs_retrieval = bool(result.needs_retrieval) if result.needs_retrieval is not None else False
    next_step = "query_understanding" if needs_retrieval else "chat_answer"
    retrieval_strategy = "hybrid"
    if result.intent_type == "navigational":
        retrieval_strategy = "bm25"
    elif result.intent_type == "explanatory":
        retrieval_strategy = "hybrid"

    return {
        "agent_log": (
            f"意图识别完成，类型: {result.intent_type}，置信度: {result.confidence:.2f}，"
            f"需要检索: {needs_retrieval}，分析: {result.analysis}"
        ),
        "intent_type": result.intent_type,
        "intent_confidence": result.confidence,
        "needs_retrieval": needs_retrieval,
        "next_step": next_step,
        "retrieval_strategy": retrieval_strategy,
    }


# 闲聊/功能咨询节点：不需要检索知识库，直接对话回答并结束
async def chat_answer_node(state: KnowledgeBaseState, writer: StreamWriter) -> KnowledgeBaseState:
    """闲聊/功能咨询节点：agent 绑定知识库内容查询工具，模型按需 function call 查库，
    流式输出：token 聚合约每 10 字或 1 秒通过 writer 推送给前端 SSE。
    """
    llm = get_model('Specialist')
    query = state.get("question", "")
    scope = kb_scope(state)
    agent = create_workflow_agent(
        model=llm,
        node_name="chat_answer_node",
        thread_id=state.get("threadId") or "",
        user_id=state.get("userId") or "",
        system_prompt=get_prompt('knowledge', 'chat_answer'),
        tools=[query_kb_documents(scope), query_kb_tables(scope), get_current_time],
    )

    messages: List[BaseMessage] = [
        *(state.get("messages") or []),
        HumanMessage(content=query),
    ]

    answer = await agent_stream_answer(
        agent, messages, writer,
        node_name="chat_answer", start_msg="正在回复...",
    )
    answer = answer.strip() or "抱歉，暂时无法回答，请换个问法试试。"

    return {
        "agent_log": f"闲聊/功能咨询回答完成，问题: {query}",
        "answer": answer,
        "has_reliable_source": False,
        "sources": [],
    }


# 检索 agent 节点：只查证据不回答，工具结果逐步推送进度（边检索边输出过程状态）
async def retrieval_agent_node(state: KnowledgeBaseState, writer: StreamWriter) -> KnowledgeBaseState:
    """检索 agent：绑定 文档检索/表查询/目录/时间 工具，LLM 自主决定查询过程。

    与回答节点解耦（拆分自原单 agent 主查询节点）：
    - 只负责通过 tool calling 收集证据，**不回答用户问题**（回答交给 answer 节点）。
    - 每个工具结果到达即通过 writer 推送"已获取：xxx"进度，前端实时看到检索过程。
    - 工具命中来源收集到 retrieved_docs；工具结果全文汇总为 context（evidence）写入 state，
      由 answer 节点读取生成回答，上下文显式传递、无关联问题。
    - 支持多轮检索（换词重试）；结构化数据问题走确定性表工具（MongoDB 精确筛选，零幻觉）。
    """
    llm = get_model('Specialist')
    query = state.get("question", "")
    scope = kb_scope(state)
    kb_id = scope.get("kb_id")
    owner_id = scope.get("owner_id")

    # 检索来源收集：工具调用中命中的文档元数据，供 format_response 提取 sources
    retrieved_docs: List[Dict[str, Any]] = []

    agent = create_workflow_agent(
        model=llm,
        node_name="retrieval_agent",
        thread_id=state.get("threadId") or "",
        user_id=state.get("userId") or "",
        system_prompt=get_prompt('knowledge', 'retrieval_agent'),
        tools=[
            build_doc_search_tool(state, retrieved_docs),
            build_list_documents_tool(scope),
            *build_table_tools(kb_id, owner_id),
            get_current_time,
        ],
    )

    messages: List[BaseMessage] = [
        *(state.get("messages") or []),
        HumanMessage(content=query),
    ]

    # 边检索边推送进度：updates 模式逐步产出新消息，工具结果（ToolMessage）到达即推"已获取"状态
    writer({"node": "retrieval_agent", "messages": "正在检索知识库..."})
    all_messages: List[BaseMessage] = []
    try:
        async for event in agent.astream({"messages": messages}, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for _sub_node, update in event.items():
                new_msgs = update.get("messages") if isinstance(update, dict) else None
                if not new_msgs:
                    continue
                for m in new_msgs:
                    all_messages.append(m)
                    if getattr(m, "type", "") != "tool":
                        continue
                    preview = str(getattr(m, "content", "") or "").strip().replace("\n", " ")[:60]
                    if preview:
                        writer({"node": "retrieval_agent", "messages": f"已获取：{preview}"})
    except Exception as e:
        logger.warning("检索 agent 运行异常: %s", e, exc_info=True)

    # 工具结果全文汇总为证据（context），供回答节点生成答案
    context = _extract_evidence(all_messages)

    return {
        "agent_log": f"检索 agent 完成，来源文档 {len(retrieved_docs)} 个，证据长度 {len(context)} 字符",
        "context": context,
        "retrieved_docs": retrieved_docs,
    }


def _extract_evidence(messages: List[BaseMessage]) -> str:
    """把 agent 会话中的工具结果（ToolMessage）拼接为证据文本，供回答节点使用。"""
    blocks = []
    for m in messages:
        if getattr(m, "type", "") != "tool":
            continue
        content = str(getattr(m, "content", "") or "").strip()
        if content:
            blocks.append(content)
    return "\n\n".join(blocks)


# 回答节点：基于检索证据流式生成最终答案（与检索解耦，输出预算独立）
async def answer_node(state: KnowledgeBaseState, writer: StreamWriter) -> KnowledgeBaseState:
    """回答节点：读取检索节点收集的证据（state.context），流式生成最终答案。

    - 不绑定检索/表工具（时间工具除外），纯生成；使用 kb_answer 模型（max_tokens=4096），
      检索与回答输出预算互不挤压，长回答不再被截断。
    - agent_stream_answer 边生成边推送增量（SSE 分段回传），与检索阶段的过程状态衔接。
    - 无直接答案时基于已有资料推断总结（提示词强制标注推断）；完全没有则如实告知并给建议。
    """
    llm = get_model('kb_answer')
    query = state.get("question", "")
    context = state.get("context") or ""

    user_content = f"用户问题：{query}\n\n检索到的资料：\n{context or '（未检索到任何资料）'}"
    messages: List[BaseMessage] = [
        *(state.get("messages") or []),
        HumanMessage(content=user_content),
    ]

    agent = create_workflow_agent(
        model=llm,
        node_name="answer",
        thread_id=state.get("threadId") or "",
        user_id=state.get("userId") or "",
        system_prompt=get_prompt('knowledge', 'answer'),
        tools=[get_current_time],
    )

    answer = await agent_stream_answer(
        agent, messages, writer,
        node_name="answer", start_msg="正在生成回答...",
    )
    if not answer:
        answer = "抱歉，暂时无法回答您的问题，请换个问法试试。"

    return {
        "agent_log": f"回答完成，问题: {query}",
        "answer": answer,
    }


async def confidence_evaluation_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    """置信度评估节点：基于主查询 agent 的检索上下文评估回答可靠性。"""

    class ConfidenceOutput(BaseModel):
        confidence_score: float = Field(description="置信度分数 0-1")
        has_reliable_source: bool = Field(description="是否达到可靠标准")
        analysis: str = Field(description="评估分析")
        missing_points: List[str] = Field(description="缺失信息点")

    llm = get_model('Reviewer')
    query = state.get("question", "")
    answer = state.get("answer", "")
    context = state.get("context", "")

    if not context or "未找到" in answer:
        return {
            "agent_log": "无可靠来源，置信度校验不通过",
            "confidence_score": 0.0,
            "has_reliable_source": False,
        }

    prompt_template = get_prompt('knowledge', 'confidence_evaluation')
    prompt = prompt_template.format(query=query, answer=answer, context=context)

    structured_llm = llm.with_structured_output(ConfidenceOutput)
    result: ConfidenceOutput = await structured_llm.ainvoke(prompt)

    return {
        "agent_log": f"置信度评估完成，分数: {result.confidence_score:.2f}，可靠来源: {result.has_reliable_source}，分析: {result.analysis}",
        "confidence_score": result.confidence_score,
        "has_reliable_source": result.has_reliable_source,
    }


async def format_response_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    """格式化输出节点：从检索 agent 的记录提取 sources（参考来源）。

    注意：answer 已由 answer 节点生成并经 SSE 流式推送，
    此处**不再返回 answer**，只补 sources 等轻量参考信息，
    避免 step_format 事件重复携带完整回答导致 SSE payload 超限（>4096 字节被截断）。
    """
    # 主查询 agent 在工具调用中命中的来源（doc_name/section/score），去重后作为 sources
    raw_docs = state.get("retrieved_docs") or []
    sources = []
    if raw_docs:
        seen = set()
        for d in raw_docs:
            name = d.get("doc_name") or "未知文档"
            if name in seen:
                continue
            seen.add(name)
            sources.append({
                "doc_name": name,
                "section": d.get("section", ""),
                "score": float(d.get("score") or 0),
                "doc_id": d.get("doc_id", ""),
            })

    return {
        "agent_log": f"本次查询完成，共 {len(sources)} 个来源文档",
        "sources": sources,
    }
