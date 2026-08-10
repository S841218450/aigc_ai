"""
对话历史纯函数工具
====================
- parse_chat_history：把 Java 传入的历史消息解析为统一的 [{role, content}, ...]
- format_history_msgs：把消息列表格式化为可注入摘要的文本
- to_langchain_messages：把记忆上下文（摘要 + 窗口）转换为 LangChain 消息列表
"""
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def parse_chat_history(items: list) -> list:
    """
    把 Java 传入的历史消息解析为统一的 [{role, content}, ...]。
    兼容两种格式：
    1. 原生记录 {question, answer, ...}：一条记录 = user(question) + assistant(answer)。
       answer 为空代表"该轮尚未回答"（即最刚发出的消息，其 question 已由 query 字段单独传入）→ 整条跳过，避免与 query 重复注入
    2. 精简格式 {role, content}
    """
    msgs = []
    for item in items:
        it = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else dict(item))
        # 原生格式：通过 question/answer 区分 role
        if it.get("question") is not None or it.get("answer") is not None:
            answer = (it.get("answer") or "").strip()
            if not answer:
                continue  # 当前进行中消息（answer 为空），跳过
            question = (it.get("question") or "").strip()
            if question:
                msgs.append({"role": "user", "content": question})
            msgs.append({"role": "assistant", "content": answer})
        # 精简格式
        else:
            role = (it.get("role") or "").lower()
            content = (it.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
    return msgs


def format_history_msgs(msgs: list) -> str:
    """
    把消息列表格式化为可注入摘要的文本。
    兼容两种输入：已解析的 [{role, content}] 或 Java 原生记录 {question, answer}（answer 为空则跳过）
    """
    lines = []
    for m in msgs:
        it = m.model_dump() if hasattr(m, "model_dump") else (m if isinstance(m, dict) else dict(m))
        if it.get("question") is not None or it.get("answer") is not None:
            answer = (it.get("answer") or "").strip()
            if not answer:
                continue  # 未回答的当前消息，跳过
            question = (it.get("question") or "").strip()
            if question:
                lines.append(f"用户: {question}")
            lines.append(f"助手: {answer}")
        else:
            role = "用户" if it.get("role") == "user" else "助手"
            lines.append(f"{role}: {it.get('content', '')}")
    return "\n".join(lines)


def to_langchain_messages(summary: str, window: list) -> List[BaseMessage]:
    """把记忆上下文转换为 LangChain 消息列表（摘要置顶 + 最近窗口）。
    在初始化时一次性转换，节点直接复用，避免每次 LLM 调用重复组装。"""
    messages: List[BaseMessage] = []
    summary = (summary or "").strip()
    if summary:
        messages.append(SystemMessage(content=f"【历史摘要】{summary}"))
    for m in window:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages
