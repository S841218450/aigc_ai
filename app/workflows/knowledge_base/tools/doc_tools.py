# -*- coding: utf-8 -*-
"""知识库文档检索工具集：agent 通过 tool calling 查询文档的入口。

按用途分层：
- kb_scope / merge_state_filter_to_params：检索权限与过滤器辅助
- build_doc_search_tool：混合检索（语义+BM25），命中片段收集到调用方传入的 retrieved_docs
- build_list_documents_tool：知识库文档清单
"""
from typing import Any, Dict, List

from app.tools.knowledge_tools.search_tools import query_kb_documents
from app.tools.retrieval.vector_store import VectorStoreTool
from app.workflows.knowledge_base.state import KnowledgeBaseState


def kb_scope(state: KnowledgeBaseState) -> Dict[str, Any]:
    """提取当前检索范围（与向量检索同一套权限隔离）。"""
    params = state.get("params", {}) or {}
    return {
        "kb_id": state.get("kb_id") or params.get("kb_id"),
        "owner_id": state.get("userId") or "",
        "folder_ids": state.get("filter_folder_ids"),
        "doc_ids": state.get("filter_doc_ids"),
    }


def merge_state_filter_to_params(state: KnowledgeBaseState) -> Dict[str, Any]:
    """把 state 中独立的 kb_id / filter_folder_ids / filter_doc_ids
    合并进 params["filter"]，与 Chroma 原生 where filter 语法对齐。"""
    params = dict(state.get("params", {}) or {})
    f = dict(params.get("filter") or {})

    kb_id = state.get("kb_id") or params.get("kb_id")
    folder_ids = state.get("filter_folder_ids")
    doc_ids = state.get("filter_doc_ids")

    if kb_id and "kb_id" not in f:
        f["kb_id"] = kb_id

    if folder_ids and "folder_id" not in f:
        if len(folder_ids) == 1:
            f["folder_id"] = folder_ids[0]
        else:
            f["folder_id"] = {"$in": list(folder_ids)}

    if doc_ids and "doc_id" not in f:
        if len(doc_ids) == 1:
            f["doc_id"] = doc_ids[0]
        else:
            f["doc_id"] = {"$in": list(doc_ids)}

    if f:
        params["filter"] = f
    return params


def build_doc_search_tool(state: KnowledgeBaseState, retrieved_docs: List[Dict[str, Any]]):
    """构建「混合检索文档」工具闭包（绑定 state 过滤范围）。

    retrieved_docs 为调用方持有的列表：每次命中追加 {doc_name, section, content, score}，
    供后处理节点（format_response）提取 sources。
    """
    async def search_knowledge(query: str, top_k: int = 10) -> str:
        """在当前知识库文档中做语义+关键词混合检索，返回最相关的文档片段（含来源文档名、章节）。回答需要查文档的问题时先用它。"""
        try:
            tool = VectorStoreTool()
            search_params = merge_state_filter_to_params(state)
            search_params = {**search_params, "k": max(1, min(int(top_k or 10), 50))}
            docs = await tool.search(query, search_params)
        except Exception as e:
            return f"检索失败: {e}"
        if not docs:
            return "（未检索到相关内容）"
        lines = []
        for i, d in enumerate(docs, 1):
            md = d.get("metadata", {}) or {}
            source = md.get("doc_name") or md.get("source") or "未知来源"
            section = md.get("section") or md.get("chapter") or ""
            header = f"[{i}] 来源: {source}" + (f" 章节: {section}" if section else "")
            retrieved_docs.append({
                "doc_name": source,
                "section": section,
                "content": d.get("content", ""),
                "score": float(d.get("rerank_score") or d.get("hybrid_score") or d.get("score") or 0),
            })
            lines.append(f"{header}\n{d['content']}")
        return "\n\n".join(lines)
    return search_knowledge


def build_list_documents_tool(scope: Dict[str, Any]):
    """构建「列出文档清单」工具闭包（绑定检索 scope）。"""
    async def list_documents() -> str:
        """列出当前知识库已上传的文档清单（文档名列表）。不确定知识库有什么内容时先调用确认范围。"""
        try:
            return await query_kb_documents(scope)()
        except Exception as e:
            return f"查询文档清单失败: {e}"
    return list_documents
