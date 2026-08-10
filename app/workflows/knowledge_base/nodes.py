import json
from copy import deepcopy
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.core.agents.model_factory import get_model
from app.core.prompts.prompts_factory import get_prompt
from app.workflows.knowledge_base.state import KnowledgeBaseState
from app.tools.common.table_registry import TABLE_REGISTRY
from app.tools.retrieval.vector_store import VectorStoreTool


# ---------------------------------------------------------------------------
# 工具：把 state 中独立的 kb_id / filter_folder_ids / filter_doc_ids
#       合并进 params["filter"]，与 Chroma 原生 where filter 语法对齐
# ---------------------------------------------------------------------------
def _format_chat_history(state: KnowledgeBaseState, max_entries: int = 8) -> str:
    """把对话历史格式化为注入 prompt 的文本（摘要置顶 + 最近窗口），无历史时给占位"""
    summary = (state.get("conversation_summary") or "").strip()
    history = state.get("chat_history") or []

    blocks: List[str] = []
    if summary:
        blocks.append(f"【历史摘要】{summary}")

    lines = []
    for m in history[-max_entries:]:
        role = "用户" if m.get("role") == "user" else "助手"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    if lines:
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else "（无历史对话）"


def _merge_state_filter_to_params(state: KnowledgeBaseState) -> Dict[str, Any]:
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


class IntentRecognitionOutput(BaseModel):
    intent_type: str = Field(description="意图类型")
    confidence: float = Field(description="置信度 0-1")
    analysis: str = Field(description="分析说明")
    needs_retrieval: bool = Field(description="是否需要检索知识库；闲聊/问候/系统功能咨询等与知识库无关的问题为 False")


# 意图识别节点：根据用户查询识别意图类型，并判断是否需要知识库检索
async def intent_recognition_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    llm = get_model('intent')
    query = state.get("question", state.get("question", ""))
    prompt_template = get_prompt('knowledge', 'intent_recognition')
    prompt = prompt_template.format(query=query)

    structured_llm = llm.with_structured_output(IntentRecognitionOutput)
    result: IntentRecognitionOutput = await structured_llm.ainvoke(prompt)

    needs_retrieval = True if result.needs_retrieval is None else bool(result.needs_retrieval)
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


def kb_needed_router(state: KnowledgeBaseState) -> str:
    """意图识别后路由：不需要知识库检索（闲聊/功能咨询）直接 chat_answer 结束；
    结构化类问题（枚举/计数/筛选/推荐）走结构化查询节点；其余走向量检索链路。"""
    if state.get("needs_retrieval", True) is False:
        return "chat_answer"
    if state.get("intent_type") == "structured":
        return "structured_query"
    return "query_understanding"


# 闲聊/功能咨询节点：不需要检索知识库，直接对话回答并结束
async def chat_answer_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    llm = get_model('Specialist')
    query = state.get("question", "")

    system_prompt = get_prompt('knowledge', 'chat_answer')
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *(state.get("messages") or []),
        HumanMessage(content=query),
    ]

    result = await llm.ainvoke(messages)
    answer = result.content.strip()

    return {
        "agent_log": f"闲聊/功能咨询回答完成，问题: {query}",
        "answer": answer,
        "has_reliable_source": False,
        "sources": [],
    }


class StructuredQueryOutput(BaseModel):
    """结构化查询规划输出：选表 + 操作 + 过滤条件（确定性执行，不靠语义检索猜）。"""
    table_id: str = Field(description="所选表的 table_id（必须原样复制自表目录）")
    op: str = Field(description="操作类型：count（统计行数）/ query（明细行）/ distinct（某列去重枚举）/ aggregate（分组统计）")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="过滤条件，键为列名；数值范围用 max_/min_ 前缀；文本模糊用 ~关键词")
    # query
    order_by: Optional[str] = Field(default=None, description="query 排序列名（数值列）")
    order_dir: str = Field(default="asc", description="query 排序方向 asc/desc")
    top_n: int = Field(default=20, description="query 返回行数上限（1-500，超过 100 条请确认问题确需大样本）")
    # distinct：某列去重枚举（"产品有哪些种类/品牌/产地"）
    distinct_column: Optional[str] = Field(default=None, description="distinct 操作时指定要去重的列名")
    distinct_limit: int = Field(default=200, description="distinct 展示上限（最多 1000 个取值）")
    # aggregate：分组统计（"每个种类各有多少个产品"）
    group_by: Optional[str] = Field(default=None, description="aggregate 分组列名")
    agg_op: str = Field(default="count", description="聚合函数：count/sum/avg/max/min")
    agg_column: Optional[str] = Field(default=None, description="sum/avg/max/min 时的聚合列名；count 留空")
    agg_limit: int = Field(default=200, description="aggregate 返回分组数上限（最多 1000）")
    reasoning: str = Field(description="选表与操作/条件设计说明")


def _format_conditions_desc(conditions: Dict[str, Any]) -> str:
    return "、".join(f"{k}={v}" for k, v in conditions.items())


# 结构化查询节点：枚举/计数/筛选/推荐类问题确定性查表（"现在有几个产品"精确回答）
async def structured_query_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    """
    1. 加载表目录（kb_id/owner_id 与向量检索同一套权限隔离）
    2. LLM 一次调用完成选表 + 条件设计（表目录体积小，几十张表几 KB）
    3. 确定性执行 count_rows / query_rows → 结果组装为 context 直接进入答案生成
    失败（无表/无效 table_id/执行异常）→ structured_failed=True 回退向量检索链路
    """
    llm = get_model('intent')
    query = state.get("question", "")
    params = state.get("params", {}) or {}
    filter_ = params.get("filter") or {}

    tables = await TABLE_REGISTRY.list_tables(
        kb_id=state.get("kb_id") or params.get("kb_id"),
        owner_id=state.get("userId") or filter_.get("owner_id"),
        limit=200,
    )
    if not tables:
        return {
            "agent_log": "结构化查询：知识库无可用结构化表，回退向量检索",
            "structured_failed": True,
        }

    catalog_lines = []
    for t in tables:
        cols = "、".join(t.get("raw_columns") or t.get("columns") or [])
        catalog_lines.append(
            f"- table_id: {t.get('table_id')} | 摘要: {t.get('summary')} "
            f"| 列: {cols} | 行数: {t.get('row_count')}"
        )
    prompt = get_prompt('knowledge', 'structured_query').format(
        query=query,
        table_catalog="\n".join(catalog_lines),
    )

    structured_llm = llm.with_structured_output(StructuredQueryOutput)
    try:
        out: StructuredQueryOutput = await structured_llm.ainvoke(prompt)
    except Exception as e:
        return {
            "agent_log": f"结构化查询：LLM 规划失败，回退向量检索（{e}）",
            "structured_failed": True,
        }

    table = await TABLE_REGISTRY.get_table(out.table_id)
    if not table:
        return {
            "agent_log": f"结构化查询：LLM 返回无效 table_id={out.table_id}，回退向量检索",
            "structured_failed": True,
        }

    conditions = {k: v for k, v in (out.conditions or {}).items() if v is not None and v != ""}
    doc_name = table.get("doc_name", "")
    sheet_name = table.get("sheet_name", "")
    raw_columns = table.get("raw_columns") or table.get("columns") or []
    op = (out.op or "query").lower().strip()

    try:
        cond_desc = _format_conditions_desc(conditions)

        if op == "count":
            n = await TABLE_REGISTRY.count_rows(out.table_id, conditions)
            result_text = (
                f"表「{doc_name}({sheet_name})」统计结果：共 {n} 条"
                + (f"（条件：{cond_desc}）" if cond_desc else "")
            )
            log_extra = f"count={n}"

        elif op == "distinct":
            if not out.distinct_column:
                raise ValueError("distinct 操作必须指定 distinct_column")
            dv = await TABLE_REGISTRY.distinct_values(
                out.table_id,
                out.distinct_column,
                conditions,
                limit=max(1, min(int(out.distinct_limit or 200), 1000)),
            )
            counts = dv.get("counts", [])
            lines = [f"表「{doc_name}({sheet_name})」中「{dv.get('column')}」去重枚举共 {dv.get('total_distinct')} 类"
                     + (f"（条件：{cond_desc}）" if cond_desc else "")
                     + "，按出现次数降序："]
            for i, item in enumerate(counts, 1):
                lines.append(f"{i}. {item.get('value')}（{item.get('count')} 条）")
            result_text = "\n".join(lines)
            log_extra = f"distinct_col={dv.get('column')} distinct_cnt={dv.get('total_distinct')}"

        elif op == "aggregate":
            if not out.group_by:
                raise ValueError("aggregate 操作必须指定 group_by")
            if out.agg_op in ("sum", "avg", "max", "min") and not out.agg_column:
                raise ValueError(f"aggregate {out.agg_op} 必须指定 agg_column")
            agg = await TABLE_REGISTRY.aggregate_stats(
                out.table_id,
                out.group_by,
                conditions,
                agg_column=out.agg_column,
                agg_op=out.agg_op or "count",
                limit=max(1, min(int(out.agg_limit or 200), 1000)),
            )
            groups = agg.get("groups", [])
            agg_col_txt = f"/聚合列:{agg.get('agg_column')}" if agg.get("agg_column") else ""
            lines = [f"表「{doc_name}({sheet_name})」按「{agg.get('group_by')}」{agg.get('agg_op')}"
                     + agg_col_txt
                     + (f"（条件：{cond_desc}）" if cond_desc else "")
                     + f"共 {agg.get('total_groups')} 组，按统计值降序："]
            for i, item in enumerate(groups, 1):
                lines.append(f"{i}. {item.get('group')}: {item.get('value')}")
            result_text = "\n".join(lines)
            log_extra = (
                f"group_by={agg.get('group_by')} op={agg.get('agg_op')} "
                f"groups={agg.get('total_groups')}"
            )

        else:  # query：明细行（LLM 自主决定 top_n，上限 500）
            res = await TABLE_REGISTRY.query_rows(
                out.table_id,
                conditions,
                top_n=max(1, min(int(out.top_n or 20), 500)),
                order_by=out.order_by,
                order_dir=out.order_dir,
            )
            rows = res.get("rows", [])
            lines = [f"表「{doc_name}({sheet_name})」满足条件共 {res.get('total_matched')} 条，展示前 {len(rows)} 条"
                     + (f"（条件：{cond_desc}）" if cond_desc else "") + "："]
            for i, r in enumerate(rows, 1):
                data = r.get("data", {})
                pairs = "，".join(f"{c}: {data.get(c, '')}" for c in raw_columns if c in data)
                lines.append(f"{i}. {pairs}")
            result_text = "\n".join(lines)
            log_extra = f"matched={res.get('total_matched')} returned={len(rows)}"
    except Exception as e:
        return {
            "agent_log": f"结构化查询：执行失败（op={op} err={e}），回退向量检索",
            "structured_failed": True,
        }

    return {
        "agent_log": f"结构化查询完成 op={op} {log_extra}，table_id={out.table_id}，reasoning={out.reasoning}",
        "structured_context": result_text,
        "context": result_text,
        "has_reliable_source": True,
        "structured_failed": False,
    }


def structured_router(state: KnowledgeBaseState) -> str:
    """结构化查询后路由：成功 → 直接生成答案（跳过向量检索链路）；失败 → 回退查询理解。"""
    if state.get("structured_failed"):
        return "query_understanding"
    return "generate_answer"

# 查询重写节点：把用户口语化表达的查询转换为结构化查询给【检索模型】使用
async def query_rewrite_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    llm = get_model('Specialist')
    query = state.get("question", state.get("question", ""))
    intent_type = state.get("intent_type", "factual")
    prompt_template = get_prompt('knowledge', 'query_rewrite')
    prompt = prompt_template.format(
        query=query,
        intent_type=intent_type,
        chat_history=_format_chat_history(state),
    )

    result = await llm.ainvoke(prompt)
    rewritten_query = result.content.strip()

    return {
        "agent_log": f"查询优化完成，原始查询: {query}，优化后: {rewritten_query}",
        "query_rewritten": rewritten_query,
    }

# 实体提取节点：从查询中提取实体和关系，例如“北京到上海的航班”中的“北京”、“上海”、“航班”等
async def entity_extraction_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    class EntityOutput(BaseModel):
        entities: List[Dict[str, str]] = Field(description="实体列表")
        relations: List[Dict[str, str]] = Field(description="关系列表")

    llm = get_model('intent')
    query = state.get("query_rewritten") or state.get("question", state.get("question", ""))
    intent_type = state.get("intent_type", "factual")
    prompt_template = get_prompt('knowledge', 'entity_extraction')
    prompt = prompt_template.format(query=query, intent_type=intent_type)

    structured_llm = llm.with_structured_output(EntityOutput)
    result: EntityOutput = await structured_llm.ainvoke(prompt)

    return {
        "agent_log": f"实体抽取完成，提取到 {len(result.entities)} 个实体，{len(result.relations)} 个关系",
        "kg_entities": result.entities,
    }


class QueryUnderstandingOutput(BaseModel):
    rewritten_query: str = Field(description="改写优化后的检索查询")
    entities: List[Dict[str, str]] = Field(description="抽取到的实体列表，每项包含 name 和 type 字段")
    relations: List[Dict[str, str]] = Field(description="抽取到的实体关系列表")

# 查询理解节点（合并 重写 + 实体抽取，一次 LLM 调用双字段输出）
async def query_understanding_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    llm = get_model('intent')
    query = state.get("question", state.get("question", ""))
    intent_type = state.get("intent_type", "factual")

    system_prompt = get_prompt('knowledge', 'query_understanding')

    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *(state.get("messages") or []),
        HumanMessage(content=f"原始问题：{query}\n意图类型：{intent_type}"),
    ]

    structured_llm = llm.with_structured_output(QueryUnderstandingOutput)
    result: QueryUnderstandingOutput = await structured_llm.ainvoke(messages)

    rewritten_query = (result.rewritten_query or query).strip() or query

    return {
        "agent_log": (
            f"查询理解完成，原始查询: {query}，"
            f"改写后: {rewritten_query}，"
            f"抽取到 {len(result.entities)} 个实体，{len(result.relations)} 个关系"
        ),
        "query_rewritten": rewritten_query,
        "kg_entities": result.entities,
    }


# BM25检索节点：根据查询进行BM25关键词检索，返回相关文档
async def bm25_retrieval_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    tool = VectorStoreTool()
    query = state.get("query_rewritten") or state.get("question", state.get("question", ""))
    params = _merge_state_filter_to_params(state)

    search_params = {**params, "k": params.get("bm25_k", 50)}
    docs = await tool.bm25_search(query, search_params)

    return {
        "agent_log": f"BM25 关键词检索完成，召回 {len(docs)} 篇文档"
                     + (f"（filter={params.get('filter')}）" if params.get("filter") else ""),
        "bm25_docs": docs,
    }


async def vector_retrieval_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    """向量检索节点"""
    tool = VectorStoreTool()
    query = state.get("query_rewritten") or state.get("question", state.get("question", ""))
    params = _merge_state_filter_to_params(state)

    search_params = {**params, "k": params.get("vector_k", 50)}
    docs = await tool.vector_search(query, search_params)

    return {
        "agent_log": f"向量检索完成，召回 {len(docs)} 篇文档"
                     + (f"（filter={params.get('filter')}）" if params.get("filter") else ""),
        "vector_docs": docs,
    }


async def hybrid_merge_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    """合并BM25和向量检索结果"""
    tool = VectorStoreTool()
    query = state.get("query_rewritten") or state.get("question", state.get("question", ""))
    params = state.get("params", {}) or {}
    strategy = state.get("retrieval_strategy", "hybrid")

    if strategy == "bm25":
        docs = state.get("bm25_docs", [])
    elif strategy == "vector":
        docs = state.get("vector_docs", [])
    else:
        bm25_docs = state.get("bm25_docs", [])
        vector_docs = state.get("vector_docs", [])

        vector_weight = params.get("vector_weight", 0.6)
        bm25_weight = params.get("bm25_weight", 0.4)
        top_k = params.get("hybrid_k", 20)

        doc_map: Dict[str, Dict[str, Any]] = {}

        for doc in vector_docs:
            doc_id = doc["metadata"].get("chunk_id") or doc["content"][:80]
            normalized_score = tool._normalize_score(doc["score"], "vector")
            doc_map[doc_id] = {
                **doc,
                "hybrid_score": normalized_score * vector_weight,
                "vector_score": doc["score"],
                "bm25_score": 0.0,
                "retrieval_method": "vector",
            }

        for doc in bm25_docs:
            doc_id = doc["metadata"].get("chunk_id") or doc["content"][:80]
            normalized_score = tool._normalize_score(doc["score"], "bm25")
            if doc_id in doc_map:
                doc_map[doc_id]["hybrid_score"] += normalized_score * bm25_weight
                doc_map[doc_id]["bm25_score"] = doc["score"]
                doc_map[doc_id]["retrieval_method"] = "hybrid"
            else:
                doc_map[doc_id] = {
                    **doc,
                    "hybrid_score": normalized_score * bm25_weight,
                    "vector_score": 0.0,
                    "bm25_score": doc["score"],
                    "retrieval_method": "bm25",
                }

        docs = sorted(
            doc_map.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:top_k]

    return {
        "agent_log": f"混合检索合并完成，共 {len(docs)} 篇候选文档，检索策略: {strategy}",
        "hybrid_docs": docs,
    }


async def rerank_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    tool = VectorStoreTool()
    query = state.get("query_rewritten") or state.get("question", state.get("question", ""))
    params = state.get("params", {}) or {}
    docs = state.get("hybrid_docs", [])

    reranked_docs = await tool.rerank(query, docs, params)

    return {
        "agent_log": f"重排序完成，Top-{len(reranked_docs)} 文档筛选完毕",
        "reranked_docs": reranked_docs,
        "documents": reranked_docs,
    }


async def _load_attachment_text(att: Dict[str, Any]) -> str:
    """文档附件 → 纯文本：优先下载解析 URL，其次直接用 content 兜底"""
    url = (att.get("url") or "").strip()
    content = (att.get("content") or "").strip()
    if url:
        # 懒加载避免与 services 层循环依赖
        from app.tools.common.file_download import download_and_extract_content
        extracted, _, _, _ = await download_and_extract_content(
            url, file_name_hint=att.get("name") or ""
        )
        return extracted or ""
    return content


async def context_building_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    tool = VectorStoreTool()
    docs = state.get("reranked_docs", state.get("documents", []))

    context = tool.build_context(docs)

    # 文档附件：下载解析后追加为补充依据（不参与检索评分，仅作为答案生成的上下文）
    doc_attachments = [
        a for a in (state.get("attachments") or [])
        if a.get("type") == "document"
    ]
    extra_blocks = []
    for att in doc_attachments:
        name = att.get("name") or "附件"
        try:
            text = await _load_attachment_text(att)
        except Exception as e:
            extra_blocks.append(f"【附件文档: {name}】（解析失败: {e}）")
            continue
        if text.strip():
            extra_blocks.append(f"【附件文档: {name}】\n{text.strip()}")

    if extra_blocks:
        context = (context + "\n\n" if context else "") + "\n\n".join(extra_blocks)

    return {
        "agent_log": (
            f"上下文构建完成，共 {len(docs)} 个文档片段，"
            f"{len(extra_blocks)} 个附件文档，上下文长度: {len(context)} 字符"
        ),
        "context": context,
    }


async def generate_answer_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    """结果生成节点"""
    llm = get_model('Specialist')
    query = state.get("question", state.get("question", ""))
    context = state.get("context", "")

    # 图片附件 → 多模态（image_url 内容块），让模型真正看图
    images = [
        a for a in (state.get("attachments") or [])
        if a.get("type") == "image"
    ]

    if not context and not images:
        return {
            "agent_log": "无检索到的文档，返回未找到依据",
            "answer": "未找到可靠依据，请尝试调整您的问题关键词。",
            "confidence_score": 0.0,
            "has_reliable_source": False,
        }

    prompt_template = get_prompt('knowledge', 'answer_generation')
    user_content = f"用户问题：{query}\n\n检索到的文档片段：\n{context}"
    # 历史对话
    messages: List[BaseMessage] = [
        SystemMessage(content=prompt_template),
        *(state.get("messages") or []),
    ]

    if images:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_content}]
        for img in images:
            img_url = (img.get("url") or "").strip()
            if not img_url:
                # 无 URL 时尝试把 content 当 data URI / base64 传给 image_url
                raw = (img.get("content") or "").strip()
                if raw:
                    img_url = raw if raw.startswith("data:") else f"data:image/png;base64,{raw}"
            if img_url:
                content.append({"type": "image_url", "image_url": {"url": img_url}})
        if len(content) == 1:
            # 图片均无可用 URL/内容，退化为纯文本
            result = await llm.ainvoke(messages + [HumanMessage(content=user_content)])
        else:
            result = await llm.ainvoke(messages + [HumanMessage(content=content)])
    else:
        result = await llm.ainvoke(messages + [HumanMessage(content=user_content)])

    answer = result.content.strip()

    return {
        "agent_log": (
            f"答案生成完成，答案长度: {len(answer)} 字符"
            + (f"，携带 {len(images)} 张图片附件" if images else "")
        ),
        "answer": answer,
    }


async def confidence_evaluation_node(state: KnowledgeBaseState) -> KnowledgeBaseState:
    """置信度评估节点"""
    class ConfidenceOutput(BaseModel):
        confidence_score: float = Field(description="置信度分数 0-1")
        has_reliable_source: bool = Field(description="是否达到可靠标准")
        analysis: str = Field(description="评估分析")
        missing_points: List[str] = Field(description="缺失信息点")

    # 结构化查询结果：确定性查表（count/query 精确执行），无需 LLM 置信度评审，直接认定可靠
    # 跳过一次 Reviewer 调用（含实例创建），省去结构化链路上一次网络往返耗时
    if state.get("structured_context"):
        return {
            "agent_log": "结构化查询结果，跳过置信度评估（确定性数据，直接认定可靠）",
            "confidence_score": 1.0,
            "has_reliable_source": True,
        }

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
    """结构化输出节点"""
    tool = VectorStoreTool()
    # 结构化查询分支不经过 rerank 节点，reranked_docs 为 None；用 or 链统一兜底空列表
    docs = state.get("reranked_docs") or state.get("documents") or []
    has_reliable = state.get("has_reliable_source", False)
    answer = state.get("answer", "")

    sources = tool.extract_sources(docs)

    if not has_reliable:
        final_answer = (
            "抱歉，未找到足够可靠的依据来回答您的问题。\n\n"
            "可能的原因：\n"
            "1. 知识库中暂无相关内容\n"
            "2. 您的问题表述可能需要调整\n"
            "3. 相关内容可能需要更高的访问权限\n\n"
            "建议：尝试使用更精确的关键词，或联系管理员获取帮助。"
        )
    else:
        final_answer = answer

    return {
        "agent_log": f"响应格式化完成，共 {len(sources)} 个来源文档",
        "sources": sources,
        "answer": final_answer,
    }


def retrieval_router(state: KnowledgeBaseState) -> str:
    strategy = state.get("retrieval_strategy", "hybrid")
    if strategy == "bm25":
        return "bm25_retrieval"
    elif strategy == "vector":
        return "vector_retrieval"
    else:
        return "hybrid_parallel"



