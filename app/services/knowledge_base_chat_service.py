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
from typing import List, Optional

from app.api.java_client import API
from app.core.agents.model_factory import get_model
from app.core.prompts.prompts_factory import get_prompt
from app.services.chat_memory import chat_memory
from app.services.event_store import event_store
from app.services.knowledge_base_service import get_chat_window, update_chat
from app.services.sse_service import SSEService
from app.utils.chat_history import format_history_msgs, to_langchain_messages
from app.utils.logger_handle import logger
from app.workflows.knowledge_base.graph import KnowledgeBaseGraph

# 滑动窗口上限（轮）。Java 现取传入超过该轮数时，超窗部分滚动进近期摘要
WINDOW_MAX_TURNS = 10


async def _roll_conversation_summary(thread_id: str, old_summary: Optional[str], overflow_msgs: list) -> Optional[str]:
    """
    把即将被窗口丢弃的早期对话增量提炼进近期摘要（Agent 自持）。
    成功返回新摘要（写库与游标推进由调用方统一完成）；失败返回 None（保留旧摘要，不推进游标，
    下次请求会重试同一段，避免关键信息永久丢失）。
    """
    if not overflow_msgs:
        return old_summary  # 空 overflow：无需提炼，直接保留旧摘要（与调用方"失败保留旧摘要"行为一致）
    try:
        llm = get_model('summarizer')
        prompt = get_prompt('knowledge', 'conv_summary').format(
            old_summary=old_summary or "（无）",
            overflow=format_history_msgs(overflow_msgs),
        )
        result = await llm.ainvoke(prompt)
        new_summary = (result.content or "").strip()
        if new_summary:
            return new_summary
    except Exception as e:
        logger.warning(f"[记忆摘要] 滚动摘要失败 thread={thread_id}: {e}")
    return None


async def _rollup_overflow(thread_id: str, summarized_upto_seq: Optional[int]) -> Optional[str]:
    """
    滚动摘要核心：从本地全量日志中挑出"已超出窗口且尚未总结"的早期消息，
    提炼合并进摘要；成功后推进游标到窗口起点。
    返回更新后的摘要（无待提炼或提炼失败时返回原摘要/None）。
    """
    logs = await chat_memory.get_logs(thread_id)
    window_msgs = WINDOW_MAX_TURNS * 2
    if len(logs) <= window_msgs:
        return None  # 日志还没积累到超出窗口，无从滚动
    # 窗口内第一条消息的 seq：seq < 它的消息都已滑出窗口
    keep_from_seq = logs[-window_msgs]["seq"]
    overflow = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in logs
        if m["seq"] > (summarized_upto_seq or 0) and m["seq"] < keep_from_seq
    ]
    if not overflow:
        return None  # 待提炼区间为空（要么全部已总结，要么没有溢出）

    old_summary = await chat_memory.get_summary(thread_id)
    new_summary = await _roll_conversation_summary(thread_id, old_summary, overflow)
    if new_summary is None:
        return old_summary  # 提炼失败：游标不动，下次重试
    # 提炼成功：写库并推进游标到窗口起点（这段历史已被摘要吸收，不再重复提炼）
    await chat_memory.update_summary(thread_id, new_summary, keep_from_seq)
    return new_summary


async def _build_context(thread_id: str, message_history: Optional[list]):
    """
    组装记忆上下文，返回 (conversation_summary, window_history, messages)：
    - 滑动窗口（Java 持久化优先）：request.chat_history 现取传入（权威源）→ 未传时主动
      get_chat_window 现取最近 10 条（5 轮，{question, answer} 转 {role, content}）→ 都没有用短期缓存兜底
    - 近期摘要：读 Agent 自持摘要；Java 传入超过窗口上限时，超窗部分滚动进摘要
    - messages：窗口历史 + 摘要 → LangChain 消息列表（初始化时一次转换，供工作流节点直接注入）
    """
    if message_history:
        window = message_history[-WINDOW_MAX_TURNS * 2:]
    else:
        window = None
        # Java 持久化历史兜底：get_chat_window 只取最近 10 条消息（5 轮），answer 为空的进行中消息会被 parse 跳过
        try:
            resp = await get_chat_window(thread_id)
            if resp.get("success") and (resp.get("data") or {}).get("list"):
                parsed = parse_chat_history(resp["data"]["list"])
                if parsed:
                    window = parsed[-WINDOW_MAX_TURNS * 2:]
        except Exception as e:
            logger.warning(f"[记忆上下文] get_chat_window 拉取失败 thread={thread_id}: {e}")
        if window is None:
            window = await chat_memory.get_messages(thread_id, max_turns=WINDOW_MAX_TURNS)

    summary, summarized_upto_seq = await chat_memory.get_summary_meta(thread_id)
    # 滚动摘要：窗口划定前，把本地日志中已滑出窗口且尚未总结的早期对话提炼进摘要
    # （本地日志自持全量，不依赖 Java 只传最近 10 轮的限制）
    rolled = await _rollup_overflow(thread_id, summarized_upto_seq)
    if rolled is not None:
        summary = rolled

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
            # 采集最终 answer：检索分支由 answer 节点输出（format_response 已不携带 answer），
            # 闲聊分支由 chat_answer 节点输出。注意不能用无差别取 answer——
            # node_start 事件里是流式增量片段，会覆盖完整答案。
            if node_name in ("answer", "chat_answer", "format_response"):
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
            await chat_memory.append_log(thread_id, "assistant", last_answer)


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
        await chat_memory.append_log(thread_id, "user", query, msg_id=message_id or None)

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


# ======================================================================
# 记忆同步回调（业务端删除会话/消息时调用，防止本地缓存残留脏数据回灌上下文）
# ======================================================================

async def clear_session_memory(thread_id: str) -> None:
    """业务端删除整个会话时回调：清空 Agent 本地该会话全部记忆缓存
    （短期窗口 / 7 天摘要 / 30 天日志）。幂等：不存在也安全。"""
    await chat_memory.clear_all(thread_id)


async def delete_session_messages(thread_id: str, message_ids: List[str]) -> dict:
    """业务端删除某段消息时回调：从本地日志删除对应消息（含紧随其后的 assistant 回答）。

    摘要一致性：若被删消息落在已总结区间（seq ≤ 摘要游标），摘要里残留被删内容，
    直接作废摘要（游标一并清除），下次请求滚动时基于剩余日志全量重建。
    返回 {matched, summary_reset} 供 Java 对账。
    """
    if not message_ids:
        return {"matched": 0, "summary_reset": False}
    matched, invalidated = await chat_memory.delete_log_messages(thread_id, message_ids)
    if invalidated:
        await chat_memory.reset_summary(thread_id)
    return {"matched": matched, "summary_reset": invalidated}


# ======================================================================
# 会话标题（新建会话时回调，异步生成后回调 Java 保存）
# ======================================================================

# 持有后台任务引用，防止任务在请求返回后（引用消失）被 GC 中断
_bg_tasks: "set" = set()


async def generate_conversation_title(thread_id: str, question: str) -> None:
    """后台任务：为新建会话生成标题并回调 Java 保存。

    调用方立即返回，本函数不阻塞用户对话；失败仅记日志，不影响主流程。
    标题生成用轻量 summarizer 模型，回调 Java 新增接口保存（幂等：仅无标题时更新）。
    """
    try:
        llm = get_model('summarizer')
        prompt = get_prompt('knowledge', 'conv_title').format(question=question or "")
        result = await llm.ainvoke(prompt)
        title = (result.content or "").strip()
        if not title:
            logger.warning(f"[会话标题] 模型未生成标题 thread={thread_id}")
            return
        title = title[:30]
        await API.post("/api/ai/session/update", {"id": thread_id, "title": title})
        logger.info(f"[会话标题] 已生成并保存 thread={thread_id} title={title}")
    except Exception as e:
        logger.warning(f"[会话标题] 生成/保存失败 thread={thread_id}: {e}")


def spawn_title_task(thread_id: str, question: str) -> None:
    """创建标题生成后台任务并持有引用（响应立即返回，不等待结果）。"""
    task = asyncio.create_task(generate_conversation_title(thread_id, question))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
