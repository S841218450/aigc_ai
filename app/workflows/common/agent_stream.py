# -*- coding: utf-8 -*-
"""agent 流式运行公共辅助：供工作流内各 agent 节点复用（聊天/主查询等）。

统一节流规则：累计满 STREAM_CHUNK_CHARS 个字符 或 距上次推送超过 STREAM_FLUSH_INTERVAL 秒，
就通过 writer 向 SSE 推送一次增量文本。
"""
import logging
import time
from typing import List

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.types import StreamWriter

logger = logging.getLogger(__name__)

STREAM_CHUNK_CHARS = 10
STREAM_FLUSH_INTERVAL = 1.0


def extract_chunk_text(msg: AIMessage) -> str:
    """提取消息块中的纯文本增量（兼容 str 与 content-block list 两种格式）。"""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("text") or c.get("content")
                if t:
                    parts.append(t)
            elif isinstance(c, str):
                parts.append(c)
        return "".join(parts)
    return ""


async def agent_stream_answer(agent, messages: List[BaseMessage], writer: StreamWriter,
                              node_name: str, start_msg: str) -> str:
    """agent 流式运行（stream_mode=messages）+ 节流推送增量文本；返回最终完整回答。

    工具调用轮（chunk.tool_calls 非空）无回答文本，跳过。
    """
    writer({"node": node_name, "messages": start_msg})

    delta_buffer: List[str] = []
    last_push = time.monotonic()
    full_answer = ""

    async def push_delta():
        nonlocal last_push
        text = "".join(delta_buffer)
        if not text:
            return
        delta_buffer.clear()
        last_push = time.monotonic()
        writer({
            "node": node_name,
            "status": "正在回复",
            "streaming": True,
            "answer": text,
        })

    try:
        async for event in agent.astream({"messages": messages}, stream_mode="messages"):
            if not isinstance(event, tuple):
                continue
            chunk, _ = event
            if not isinstance(chunk, AIMessage):
                continue
            if getattr(chunk, "tool_calls", None):
                # 工具调用轮：无最终回答文本，跳过
                continue
            text = extract_chunk_text(chunk)
            if not text:
                continue
            full_answer += text
            delta_buffer.append(text)
            if len("".join(delta_buffer)) >= STREAM_CHUNK_CHARS or time.monotonic() - last_push >= STREAM_FLUSH_INTERVAL:
                await push_delta()
    except Exception as e:
        logger.warning("%s 流式回答异常: %s", node_name, e, exc_info=True)
    # 流结束：flush 剩余增量
    await push_delta()
    return full_answer
