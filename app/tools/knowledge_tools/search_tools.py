from typing import Dict, Any

from langchain_core.messages import AIMessage

from app.services.knowledge_base_service import KB_METADATA_SERVICE
from app.tools.common.table_registry import TABLE_REGISTRY
from app.utils.logger_handle import logger


async def _list_doc_names(scope: Dict[str, Any]) -> str:
    """已就绪文档名清单（按 kb/doc_ids 范围隔离），无则返回空串。"""
    docs_result = await KB_METADATA_SERVICE.list_documents(
        kb_id=scope["kb_id"] or "default",
        status="ready",
        page=1,
        page_size=200,
    )
    items = docs_result.get("items") or []
    doc_ids = scope["doc_ids"]
    if doc_ids:
        id_set = set(doc_ids)
        items = [d for d in items if d.get("doc_id") in id_set]
    names = [d.get("doc_name", "").strip() for d in items if d.get("doc_name")]
    return "\n".join(names[:50]) if names else ""


async def _list_table_summaries(scope: Dict[str, Any]) -> str:
    """结构化数据表摘要清单（来源文档、Sheet、列名样例），无则返回空串。"""
    folder_ids = scope["folder_ids"] or []
    tables = await TABLE_REGISTRY.list_tables(
        kb_id=scope["kb_id"] or "default",
        owner_id=scope["owner_id"],
        folder_id=folder_ids[0] if len(folder_ids) == 1 else None,
        limit=50,
    )
    if not tables:
        return ""
    lines = []
    for t in tables:
        name = t.get("doc_name") or ""
        sheet = t.get("sheet_name") or ""
        summary = t.get("summary") or ""
        if len(summary) > 120:
            summary = summary[:119] + "…"
        lines.append(f"- {name}({sheet})：{summary}")
    return "\n".join(lines)


def _extract_agent_text_answer(agent_state: Dict[str, Any], fallback: str = "") -> str:
    """从 create_agent 返回状态里取最后一条助手消息的文本作为最终回答。"""
    messages = agent_state.get("messages") if isinstance(agent_state, dict) else None
    if not messages:
        return fallback
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str):
            return content.strip() or fallback
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("text")]
            text = "".join(parts).strip()
            if text:
                return text
    return fallback


def query_kb_documents(scope) -> Any:
    """生成「查询知识库文档清单」工具闭包（绑定检索 scope）。"""
    async def _query_kb_documents() -> str:
        """查询当前知识库已就绪的文档清单（文档名列表）。用户询问"你能做什么/知识库有哪些内容/有哪些文档"时调用。"""
        try:
            return await _list_doc_names(scope)
        except Exception as e:
            logger.warning("查询知识库文档清单失败: %s", e)
            return ""
    return _query_kb_documents


def query_kb_tables(scope) -> Any:
    """生成「查询结构化数据表」工具闭包（绑定检索 scope）。"""
    async def _query_kb_tables() -> str:
        """查询当前知识库登记的结构化数据表（来源文档、Sheet、列名与样例摘要）。用户询问"你能查什么数据/有哪些表格"时调用。"""
        try:
            return await _list_table_summaries(scope)
        except Exception as e:
            logger.warning("查询知识库数据表失败: %s", e)
            return ""
    return _query_kb_tables