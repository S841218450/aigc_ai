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
       answer 为空时**仍保留 user 消息**（如"回答的时候请在前缀称呼我为西米"这类指令
       往往只有 user 没有 assistant，若整条跳过会导致历史指令丢失），只是不追加 assistant；
       当前进行中消息（question 与当前 query 相同、answer 空）由调用方按 query 去重。
    2. 精简格式 {role, content}
    """
    msgs = []
    for item in items:
        it = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else dict(item))
        # 原生格式：通过 question/answer 区分 role
        if it.get("question") is not None or it.get("answer") is not None:
            answer = (it.get("answer") or "").strip()
            question = (it.get("question") or "").strip()
            if question:
                msgs.append({"role": "user", "content": question})
            if answer:
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
    兼容两种输入：已解析的 [{role, content}] 或 Java 原生记录 {question, answer}（answer 为空时保留 user 行）
    """
    lines = []
    for m in msgs:
        it = m.model_dump() if hasattr(m, "model_dump") else (m if isinstance(m, dict) else dict(m))
        if it.get("question") is not None or it.get("answer") is not None:
            answer = (it.get("answer") or "").strip()
            question = (it.get("question") or "").strip()
            if question:
                lines.append(f"用户: {question}")
            if answer:
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
