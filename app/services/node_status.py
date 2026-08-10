"""
节点状态映射 + 类型驱动的 data 返回
每个节点定义 data_key 决定 SSE data 的格式：
  - "messages"   → {"messages": "str"}     执行摘要
  - "selectList"  → {"selectList": [...]}   人工介入选择题
  - "url"         → {"url": "str"}          生成的图片 URL
  - "answer"      → {"answer": "str"}       最终答案
  - "sources"     → {"sources": [...]}      来源文档列表
  - "confidence"  → {"confidence": {...}}   置信度信息
"""

TEXT_TO_IMAGE_NODE_MAP = {
    "desc_code_judge_node": {
        "type": "step_judge",
        "status": "正在评估用户描述",
        "data_key": "messages",
    },
    "decision_router": {
        "type": "step_decision",
        "status": "正在进行下一步决策",
        "data_key": "messages",
    },
    "supplementary_node": {
        "type": "step_supplementary",
        "status": "正在生成补充选择题",
        "data_key": "selectList",
    },
    "human_interrupt_node": {
        "type": "step_interrupt",
        "status": "等待用户补充描述",
        "data_key": "messages",
    },
    "prompt_combined_node": {
        "type": "step_prompt_combined",
        "status": "正在优化绘图提示词",
        "data_key": "messages",
    },
    "generate_image_node": {
        "type": "step_generate",
        "status": "生成图片中",
        "data_key": "url",
    },
    "summer_node": {
        "type": "step_summary",
        "status": "正在评估生成结果",
        "data_key": "messages",
    },
}


def build_node_data(node_name: str, state_update: dict, node_map: dict = None) -> dict:
    """根据节点类型提取 data，返回统一格式。支持传入自定义 node_map"""
    _map = node_map or TEXT_TO_IMAGE_NODE_MAP
    node_info = _map.get(node_name)
    if not node_info:
        return {"messages": state_update.get("agent_log", state_update.get("answer", ""))}

    data_key = node_info.get("data_key", "messages")

    if data_key == "messages":
        return {"messages": state_update.get("agent_log", state_update.get("answer", ""))}
    elif data_key == "selectList":
        return {"selectList": state_update.get("selectList", [])}
    elif data_key in ("image_url", "url", "image_list"):
        # 图片节点统一：文生图 state 字段是 image_url，图生图是 image_list
        field = "image_url" if data_key != "image_list" else "image_list"
        raw_url = state_update.get(field)
        if isinstance(raw_url, list):
            urls = [u for u in raw_url if u]
        else:
            urls = [raw_url] if raw_url else []
        return {
            "messages": state_update.get("agent_log", ""),
            "url": urls,  # 兼容旧前端
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


# ---- 各场景节点映射 ----

KNOWLEDGE_BASE_NODE_MAP = {
    "intent_recognition": {
        "type": "step_intent",
        "status": "正在识别查询意图",
        "data_key": "messages",
    },
    "structured_query_node": {
        "type": "step_structured_query",
        "status": "正在检索知识库文档",
        "data_key": "messages",
    },
    "chat_answer": {
        "type": "step_chat",
        "status": "正在回复（无需检索知识库）",
        "data_key": "answer",
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
        "data_key": "answer",
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

TEXT_TO_VIDEO_NODE_MAP = {}

IMAGE_TO_IMAGE_NODE_MAP = {
    "params_filter_node": {
        "type": "step_filter",
        "status": "正在过滤敏感词与图像参数",
        "data_key": "messages",
    },
    "prompt_optimization_node": {
        "type": "step_prompt_optimize",
        "status": "正在优化绘图提示词",
        "data_key": "messages",
    },
    "generate_image_node": {
        "type": "step_generate",
        "status": "正在生成图片",
        "data_key": "image_list",
    },
    "quality_evaluation_node": {
        "type": "step_quality",
        "status": "正在评估生成质量",
        "data_key": "messages",
    },
    "summary_node": {
        "type": "step_summary",
        "status": "正在整理结果",
        "data_key": "answer",
    },
    "await_retry_node": {
        "type": "step_retry",
        "status": "节点执行失败，等待重试",
        "data_key": "messages",
    },
}
