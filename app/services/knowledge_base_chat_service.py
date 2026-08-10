"""
知识库问答会话编排服务
========================
职责（service 层，供薄 HTTP 层调用）：
- 组装记忆上下文：滑动窗口（Java 传入优先，短期缓存兜底）+ 近期摘要 + 超窗滚动摘要
- 用户消息写入短期缓存（重试/续传兜底）
- 用户数据隔离：强制以当前 userId 覆盖 params.filter.owner_id
- 运行 LangGraph 流，包装答案落库（assistant 回写会话记忆）
- 支持重试（append_user=False，避免重复写入用户消息）

调用方：app/api/v1/endpoints/knowledge_base.py
依赖：
- 纯函数工具：app/utils/chat_history.py（parse/format/to_langchain_messages）
- 记忆存储：app/services/chat_memory.py
- Graph：app/workflows/knowledge_base/graph.py
"""
import asyncio
from typing import Optional

from app.core.agents.model_factory import get_model
from app.core.prompts.prompts_factory import get_prompt
from app.services.chat_memory import chat_memory
from app.services.event_store import event_store
from app.services.knowledge_base_service import update_chat
from app.services.sse_service import SSEService
from app.utils.chat_history import format_history_msgs, to_langchain_messages
from app.utils.logger_handle import logger
from app.workflows.knowledge_base.graph import KnowledgeBaseGraph

# 滑动窗口上限（轮）。Java 现取传入超过该轮数时，超窗部分滚动进近期摘要
WINDOW_MAX_TURNS = 10


async def _roll_conversation_summary(thread_id: str, old_summary: Optional[str], overflow_msgs: list) -> Optional[str]:
    """
    把超窗历史增量合并进近期摘要（Agent 自持）。
    失败时保留旧摘要，不影响主流程。
    """
    if not overflow_msgs:
        return old_summary
    try:
        llm = get_model('summarizer')
        prompt = get_prompt('knowledge', 'conv_summary').format(
            old_summary=old_summary or "（无）",
            overflow=format_history_msgs(overflow_msgs),
        )
        result = await llm.ainvoke(prompt)
        new_summary = result.content.strip()
        if new_summary:
            await chat_memory.update_summary(thread_id, new_summary)
            return new_summary
    except Exception as e:
        logger.warning(f"[记忆摘要] 滚动摘要失败 thread={thread_id}: {e}")
    return old_summary


async def _build_context(thread_id: str, message_history: Optional[list]):
    """
    组装记忆上下文，返回 (conversation_summary, window_history, messages)：
    - 滑动窗口：Java 现取传入优先（权威源），否则短期缓存兜底
    - 近期摘要：读 Agent 自持摘要；Java 传入超过窗口上限时，超窗部分滚动进摘要
    - messages：窗口历史 + 摘要 → LangChain 消息列表（初始化时一次转换，供工作流节点直接注入）
    """
    if message_history:
        window = message_history[-WINDOW_MAX_TURNS * 2:]
    else:
        window = await chat_memory.get_messages(thread_id, max_turns=WINDOW_MAX_TURNS)

    summary = await chat_memory.get_summary(thread_id)
    if message_history and len(message_history) > WINDOW_MAX_TURNS * 2:
        overflow = message_history[:len(message_history) - WINDOW_MAX_TURNS * 2]
        summary = await _roll_conversation_summary(thread_id, summary, overflow)

    return summary, window, to_langchain_messages(summary, window)


async def _persist_graph_stream(stream, thread_id: str, sse: SSEService, message_id: str = ""):
    """
    包装 graph 流：
    - 透传节点事件
    - 记录最终 answer，graph 跑完后写入会话记忆（assistant）
    - 任务执行期间异步回写 Java 消息状态（update_chat，fire-and-forget 不阻塞 SSE）：
        成功 → status=1（完成）；异常 → status=2（失败）
    """
    last_answer = None
    completed = False
    try:
        async for node_name, state_update in stream:
            if node_name in ("format_response", "chat_answer"):
                last_answer = state_update.get("answer") or last_answer
            yield node_name, state_update
        completed = True
    except Exception as e:
        # 任务异常 → 异步回写失败状态，然后继续抛出（由 SSE 层决定如何呈现错误）
        if message_id and not sse.is_cancelled:
            asyncio.create_task(update_chat(message_id, "", status=2, errorMsg=str(e)[:500]))
        raise
    finally:
        if message_id and completed and not sse.is_cancelled:
            # 任务执行完成 → 异步回写 Java 消息状态（status=1 完成）
            asyncio.create_task(update_chat(message_id, last_answer or "", status=1))
        if completed and last_answer and not sse.is_cancelled:
            await chat_memory.append(thread_id, "assistant", last_answer)


async def run_query_stream(
    thread_id: str,
    query: str,
    params: dict,
    *,
    userId: str = "",
    kb_id: str = None,
    filter_folder_ids: list = None,
    filter_doc_ids: list = None,
    attachments: Optional[list] = None,
    append_user: bool = True,
    message_history: Optional[list] = None,
    message_id: str = "",
):
    """
    统一查询执行：
    1. 组装记忆上下文（滑动窗口 + 近期摘要，超窗时滚动摘要）
    2. （可选）记录 user 消息到短期缓存（重试/续传兜底）
    3. 跑 graph → 返回 (sse, 包装流)
    append_user=False 用于重试（避免重复写入用户消息）。
    message_id：Java 端进行中消息 ID，任务执行期间异步回写消息状态（update_chat）。
    """
    # 用户数据隔离：强制以当前 userId 覆盖 filter.owner_id。
    params = dict(params or {})
    if userId:
        params_filter = dict(params.get("filter") or {})
        params_filter["owner_id"] = userId
        params["filter"] = params_filter

    summary, window_history, messages = await _build_context(thread_id, message_history)
    if append_user:
        await chat_memory.append(thread_id, "user", query, params=params, user_id=userId or None)

    graph = KnowledgeBaseGraph()
    stream = graph.run_stream(
        query=query,
        params=params,
        thread_id=thread_id,
        user_id=userId,
        kb_id=kb_id,
        filter_folder_ids=filter_folder_ids,
        filter_doc_ids=filter_doc_ids,
        attachments=attachments or None,
        chat_history=window_history,
        conversation_summary=summary,
        messages=messages,
    )
    sse = SSEService(thread_id=thread_id, event_store=event_store)
    wrapped = _persist_graph_stream(stream, thread_id, sse, message_id=message_id)
    return sse, wrapped
