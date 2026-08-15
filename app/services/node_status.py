"""
节点状态映射 + 类型驱动的 data 返回

文生图 / 图生图共用同一份 NODE_MAP（节点名已统一，前端只认这一套）：
每个节点定义 data_key 决定 SSE data 的格式：
  - "messages"   → {"messages": "str"}     执行摘要
  - "selectList"  → {"selectList": [...]}   人工介入选择题
  - "image_list"  → {"url": [...], "imageList": [...]}  生成图片 URL（无论单张/多张统一数组）
  - "answer"      → {"answer": "str"}       最终答案
  - "sources"     → {"sources": [...]}      来源文档列表
  - "confidence"  → {"confidence": {...}}   置信度信息
"""

# ---- 文生图 / 图生图统一节点映射 ----
# name 为中文步骤名，供前端展示与错误提示，绝不透出原始节点名
NODE_MAP = {
    "input_check_node": {
        "type": "step_input_check",
        "status": "正在检查描述与图像参数",
        "name": "输入检查",
        "data_key": "messages",
    },
    "decision_node": {
        "type": "step_decision",
        "status": "正在判断是否需要补充描述",
        "name": "方案决策",
        "data_key": "messages",
    },
    "supplementary_node": {
        "type": "step_supplementary",
        "status": "正在生成补充描述选项",
        "name": "补充描述",
        "data_key": "selectList",
    },
    "interrupt_node": {
        "type": "step_interrupt",
        "status": "请补充描述信息以生成更符合要求的图片",
        "name": "补充描述",
        "data_key": "messages",
    },
    "prompt_optimize_node": {
        "type": "step_prompt_optimize",
        "status": "正在优化绘图提示词",
        "name": "提示词优化",
        "data_key": "messages",
    },
    "generate_node": {
        "type": "step_generate",
        "status": "正在生成图片",
        "name": "图片生成",
        "data_key": "image_list",
    },
    "retry_node": {
        "type": "step_retry",
        "status": "步骤执行失败，等待重试",
        "name": "重试",
        "data_key": "messages",
    },
}


def get_step_name(node_name: str, node_map: dict = None) -> str:
    """节点名 → 中文步骤名（供错误提示使用，不暴露原始节点名）"""
    info = (node_map or NODE_MAP).get(node_name)
    return (info or {}).get("name") or ""


def build_node_data(node_name: str, state_update: dict, node_map: dict = None) -> dict:
    """根据节点类型提取 data，返回统一格式。支持传入自定义 node_map"""
    _map = node_map or NODE_MAP
    node_info = _map.get(node_name)
    if not node_info:
        return {"messages": state_update.get("agent_log", state_update.get("answer", ""))}

    data_key = node_info.get("data_key", "messages")

    if data_key == "messages":
        return {"messages": state_update.get("agent_log", state_update.get("answer", ""))}
    elif data_key == "selectList":
        return {"selectList": state_update.get("selectList", [])}
    elif data_key == "image_list":
        # 图片统一数组：无论单张还是多张
        raw = state_update.get("image_list")
        urls = [u for u in raw if u] if isinstance(raw, list) else ([raw] if raw else [])
        return {
            "messages": state_update.get("agent_log", ""),
            "imageList": [{"id": "", "url": u} for u in urls],  # 统一图片数组格式
        }
    elif data_key == "answer":
        return {
            "answer": state_update.get("answer", ""),
            "messages": state_update.get("agent_log", ""),
        }
    elif data_key == "sources":
        return {
            "sources": state_update.get("sources", []),
            "answer": state_update.get("answer", ""),
            "confidence_score": state_update.get("confidence_score"),
            "has_reliable_source": state_update.get("has_reliable_source"),
            "messages": state_update.get("agent_log", ""),
        }
    elif data_key == "confidence":
        return {
            "confidence_score": state_update.get("confidence_score"),
            "has_reliable_source": state_update.get("has_reliable_source"),
            "messages": state_update.get("agent_log", ""),
        }
    else:
        return {"messages": state_update.get("agent_log", state_update.get("answer", ""))}


# ---- 知识库场景节点映射（独立场景，不使用统一 NODE_MAP） ----

KNOWLEDGE_BASE_NODE_MAP = {
    "intent_recognition": {
        "type": "step_intent",
        "status": "正在识别用户意图",
        "data_key": "messages",
    },
    "retrieval_agent": {
        "type": "step_retrieval",
        "status": "正在检索知识库",
        "data_key": "messages",
    },
    "answer": {
        "type": "step_answer",
        "status": "正在生成回答",
        "data_key": "messages",
    },
    "chat_answer": {
        "type": "step_chat",
        "status": "正在回复",
        "data_key": "messages",
    },
    "query_understanding": {
        "type": "step_query_understanding",
        "status": "正在解析并重写查询，提取关键实体",
        "data_key": "messages",
    },
    "query_rewrite": {
        "type": "step_rewrite",
        "status": "正在优化查询语句",
        "data_key": "messages",
    },
    "entity_extraction": {
        "type": "step_entity",
        "status": "正在提取关键实体",
        "data_key": "messages",
    },
    "bm25_retrieval": {
        "type": "step_bm25",
        "status": "正在执行关键词检索",
        "data_key": "messages",
    },
    "vector_retrieval": {
        "type": "step_vector",
        "status": "正在执行向量检索",
        "data_key": "messages",
    },
    "hybrid_merge": {
        "type": "step_merge",
        "status": "正在合并检索结果",
        "data_key": "messages",
    },
    "rerank": {
        "type": "step_rerank",
        "status": "正在重排序文档",
        "data_key": "messages",
    },
    "context_building": {
        "type": "step_context",
        "status": "正在构建上下文",
        "data_key": "messages",
    },
    "generate_answer": {
        "type": "step_generate",
        "status": "正在生成答案",
        "data_key": "messages",
    },
    "confidence_evaluation": {
        "type": "step_confidence",
        "status": "正在评估答案置信度",
        "data_key": "confidence",
    },
    "format_response": {
        "type": "step_format",
        "status": "正在整理最终结果",
        "data_key": "sources",
    },
}
