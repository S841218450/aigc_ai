import json as _json
import re
from typing import Type, TypeVar, Iterable, Any, Dict, List, Optional, Sequence, Callable, Union, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.utils import Input, Output
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.caches import BaseCache
from pydantic import ValidationError, BaseModel

T = TypeVar("T", bound=BaseModel)

# 类型别名重定向 —— 用延迟 import 避免 common_node.py 在无 langchain 环境下 import 时报错
try:
    from langchain.agents.middleware import AgentMiddleware  # noqa: F401
    from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: F401
    from langgraph.types import StreamWriter  # noqa: F401
    from langchain_core.stores import BaseStore  # noqa: F401
    from langchain_core.tools import BaseTool  # noqa: F401
    _HAS_LANGCHAIN = True
except Exception:  # pragma: no cover
    _HAS_LANGCHAIN = False
    AgentMiddleware = Any  # type: ignore[no-redef,misc,assignment]
    BaseCheckpointSaver = Any  # type: ignore[no-redef,misc,assignment]
    StreamWriter = Any  # type: ignore[no-redef,misc,assignment]
    BaseStore = Any  # type: ignore[no-redef,misc,assignment]
    BaseTool = Any  # type: ignore[no-redef,misc,assignment]


# ---------------- 1. 基础工具：剥掉 ```json ``` fence ----------------

_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)


def strip_code_fence(text: Any) -> str:
    """从任意输入中提取纯文本，并剥掉 Markdown code fence（```json / ```）。
    统一剥不掉或没有 fence 时返回 strip() 后的原始文本。
    """
    if text is None:
        return ""
    if isinstance(text, (dict, list, tuple)):
        try:
            text = _json.dumps(text, ensure_ascii=False)
        except Exception:
            text = str(text)
    raw = str(text).strip()
    if not raw:
        return ""
    m = _CODE_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw


def parse_json_lenient(text: Any) -> Any:
    """宽松 JSON 解析：剥 fence → loads；失败返回 None（不抛异常）。"""
    cleaned = strip_code_fence(text)
    if not cleaned:
        return None
    try:
        return _json.loads(cleaned)
    except (_json.JSONDecodeError, Exception):
        return None


# ---------------- 2. 结构化解析：文本/LLM输出 → Pydantic（带兜底 fallback） ----------------

def parse_structured_output(raw_text: Any, output_cls: Type[T], fallback: T) -> T:
    """把任意 LLM 输出文本（可带 ```json 包裹、坏 JSON、普通 dict 等）解析成指定 Pydantic 模型，解析失败返回 fallback。

    和 structured_output_invoke 配套，但它只负责"文本→对象"，不调 LLM；
    适用场景：create_agent / 多消息历史 / 流式分段结果 等非 with_structured_output 不好用的地方。
    """
    # 短路：已经是目标类型的实例直接返回
    if isinstance(raw_text, output_cls):
        return raw_text
    # 短路：已经是 dict 直接 validate
    if isinstance(raw_text, dict):
        try:
            return output_cls.model_validate(raw_text)
        except (ValidationError, Exception):
            return fallback

    data = parse_json_lenient(raw_text)
    if isinstance(data, dict):
        try:
            return output_cls.model_validate(data)
        except (ValidationError, Exception):
            return fallback
    return fallback


# ---------------- 3. LLM 调用 + 结构化（和 structured_output_invoke 一致风格） ----------------


async def structured_output_invoke(
    llm,
    prompt: str,
    output_cls: Type[T],
    fallback: T,
    *,
    callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
    config: Optional[RunnableConfig] = None,
    tags: Optional[Sequence[str]] = None,
) -> T:
    """结构化 LLM 调用的统一兜底。

    优先走 with_structured_output；失败时手动用 json.loads + Pydantic 解析；再失败返回 fallback。
    避免 LLM 返回坏 JSON 时整个节点崩溃（导致 checkpoint 无法恢复的异常态）。

    新参数（纯 LLM 节点需要差异化中间件/回调/追踪时用）：
        callbacks: 传给 LLM ainvoke 的回调列表（LLM start/end/error 回调，相当于 AgentMiddleware 的轻量版）
        config:    直接传 RunnableConfig（优先级高于 callbacks / tags 单独传）
        tags:      给这次调用打标签（tracing/日志聚合用）
    """
    # 统一构造 config（显式 config 覆盖零散 callbacks/tags）
    merged_config: RunnableConfig
    if config:
        merged_config = {**config}  # type: ignore[typeddict-item]
    else:
        merged_config = {}  # type: ignore[typeddict-item]
        if callbacks:
            merged_config["callbacks"] = list(callbacks)
        if tags:
            merged_config["tags"] = list(tags)

    # 1) with_structured_output
    try:
        structured_llm = llm.with_structured_output(output_cls)
        return await structured_llm.ainvoke(prompt, merged_config if merged_config else None)
    except Exception:
        pass

    # 2) 同一个 LLM 直接输出 JSON，然后手动解析
    try:
        raw = await llm.ainvoke(
            f"请严格输出符合以下 Pydantic 结构的 JSON，不要输出任何 JSON 以外的文字：\n"
            f"Schema: {output_cls.model_json_schema()}\n\n"
            f"原始 Prompt：\n{prompt}\n\nJSON:",
            merged_config if merged_config else None,
        )
        text = raw.content if hasattr(raw, "content") else str(raw)
        return parse_structured_output(text, output_cls, fallback)
    except (_json.JSONDecodeError, ValidationError, Exception):
        return fallback


# ---------------- 3B. 纯 LLM 节点通用模板：一次封装所有公共 boilerplate ----------------


async def run_structured_node(
    *,
    node_name: str,
    state: Dict[str, Any],
    llm,
    output_cls: Type[T],
    fallback: T,
    get_prompt: Callable[[Dict[str, Any]], str],
    # 👇 可选：节点内 StreamWriter（stream_mode="custom" 时由 LangGraph 注入），
    #    用于在模型调用前发出"节点开始"信号，SSE 可先展示进行中状态
    writer: Optional[StreamWriter] = None,
    # 👇 可选：LLM 差异化配置（不同节点打不同 tags/callbacks，相当于"不同中间件"）
    callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
    tags: Optional[Sequence[str]] = None,
    llm_config: Optional[RunnableConfig] = None,
    # 👇 可选：从 LLM 结果 -> 派生 state 字段（例如 summer_node 要算 new_redraw_count / next_prompt）
    state_transformer: Optional[Callable[[T, Dict[str, Any]], Dict[str, Any]]] = None,
    # 👇 可选：agent_log 生成函数（不传就默认写一个）
    make_agent_log: Optional[Callable[[T, Dict[str, Any]], str]] = None,
) -> Dict[str, Any]:
    """纯 LLM 结构化节点的"一行模板"，把 summer_node / desc_code_judge_node 这种典型
    "拿 state -> 拼 prompt -> 调 LLM 结构化 -> 算派生字段 -> 拼 agent_log -> clean_return"
    的 40+ 行 boilerplate 压缩成 1 次调用。

    使用示例（summer_node 风格）：

        return await run_structured_node(
            node_name="summer_node",
            state=state,
            llm=get_model("summarizer"),
            output_cls=SummaryOutput,
            fallback=SummaryOutput(...),
            get_prompt=lambda s: get_prompt("text_to_image", "imageSummaryJudge").format(
                question=s["question"], prompt=s["prompt"],
                image_url=(s.get("image_url") or [""])[0],
                redraw_count=s.get("redraw_count", 0) or 0,
            ),
            callbacks=[MyCustomCallback()],  # ✅ 每个节点可以加不同回调
            tags=["judge", "summary"],         # ✅ 每个节点打不同标签
            state_transformer=lambda r, s: {
                "redraw_count": s.get("redraw_count", 0) or 0 + 1 if r.need_redraw else s.get("redraw_count", 0) or 0,
                "prompt": f'{s["prompt"]}, {r.modify_suggest}' if r.need_redraw and r.modify_suggest else s["prompt"],
            },
            make_agent_log=lambda r, s: f"图片评估完成，匹配度：{r.match_score}/10，建议：{r.modify_suggest}",
        )
    """
    # 1) 拼 prompt（闭包里读 state，外部自己决定要 format 哪些字段）
    prompt = get_prompt(state)

    # 1.5) 节点开始信号：模型调用前发出（配合 stream_mode=["updates","custom"]）
    if writer:
        writer({"node": node_name})

    # 2) 调 LLM 结构化（带 callbacks / tags 差异化配置，等价于"每个 node 不同中间件"）
    extra_tags = [node_name] + (list(tags) if tags else [])
    result: T = await structured_output_invoke(
        llm, prompt, output_cls, fallback,
        callbacks=callbacks,
        tags=extra_tags,
        config=llm_config,
    )

    # 3) 合并输出：先把 LLM 结果 dump（挑有对应 state schema 字段的）+ state_transformer 的派生字段
    result_fields: Dict[str, Any] = {}
    if isinstance(result, BaseModel):
        # 注意：只取 output_cls 的字段名，别把 state 里的字段覆盖错
        for field_name in output_cls.model_fields.keys():
            result_fields[field_name] = getattr(result, field_name, None)
    elif isinstance(result, dict):
        result_fields.update(result)

    if state_transformer is not None:
        derived = state_transformer(result, state) or {}
        result_fields.update(derived)

    # 4) agent_log：自定义 > 默认
    if make_agent_log is not None:
        agent_log = make_agent_log(result, state)
    else:
        agent_log = f"[{node_name}] 结构化输出完成"
    result_fields.setdefault("agent_log", agent_log)

    # 5) 统一 clean_return
    return clean_return(result_fields)


# ---------------- 4. 消息格式转换：dict 历史 → LangChain 标准消息 ----------------

def messages_to_langchain(history: Iterable[Dict[str, Any]]) -> List[BaseMessage]:
    """把常见的 [{"role": "user"|"assistant", "content": "..."}] 格式
    转为 LangChain 需要的 [HumanMessage | AIMessage 列表。
    未知 role 一律按 HumanMessage 处理。
    """
    out: List[BaseMessage] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        content = msg.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role:
            # 其他 role 也转成 HumanMessage，带 original role 通过 additional_kwargs 保留供后续调试
            out.append(HumanMessage(content=content, additional_kwargs={"original_role": role}))
        else:
            out.append(HumanMessage(content=content))
    return out


# ---------------- 5. Agent 结果 → 结构化解析 ----------------

def parse_agent_result(
    agent_state: Dict[str, Any],
    output_cls: Type[T],
    fallback: T,
    *,
    pick_from: str = "last_assistant",
) -> T:
    """从 create_agent 的 return state（一般含 messages 列表）里取出目标 Pydantic 结构化输出。

    默认取最后一条 AIMessage/assistant 消息（pick_from="last_assistant"）；
    解析失败返回 fallback，不会抛异常。
    """
    messages: Optional[List[BaseMessage]] = agent_state.get("messages") if isinstance(agent_state, dict) else None
    if not messages:
        return fallback

    target_msg: Optional[BaseMessage] = None
    if pick_from == "last_assistant":
        # 从后往前找第一条 AIMessage
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                target_msg = msg
                break
        if target_msg is None and messages:
            # 兜底：没有 AIMessage 时取最后一条
            target_msg = messages[-1]
    else:
        target_msg = messages[-1] if messages else None

    if target_msg is None:
        return fallback

    # 优先：AIMessage 可能带 structured_response（create_agent + response_format 时）
    structured = getattr(target_msg, "structured_response", None)
    if isinstance(structured, output_cls):
        return structured
    # 再尝试把 content 解析成 Pydantic
    raw_text = getattr(target_msg, "content", None)
    return parse_structured_output(raw_text, output_cls, fallback)


# ---------------- 6. State 辅助：dict 字段过滤 ----------------

def clean_return(data: Dict[str, Any]) -> Dict[str, Any]:
    """节点 return state 的最终返回值清理：过滤掉 None 值字段，避免 reducer 用 None 覆盖上游有效数据。
    LangGraph 的 reducer 会把显式 return 回的 None 视为"写回 None"，容易丢前序节点数据。
    调用方：每个节点 return clean_return({ ... })
    """
    return {k: v for k, v in data.items() if v is not None}


# ---------------- 7. 选择结果格式化：selectResult → 人类可读文本 ----------------

def format_select_result(
    select_result: Iterable[Any],
    *,
    sep: str = "，",
    item_sep: str = ":",
    fallback: str = "",
) -> str:
    """把 selectResult（常见格式：[{'dimension': '风格', 'answer': '写实摄影'}, ...]，
    或纯字符串列表，或任意 dict/str 混合）转成适合拼接进 prompt 的中文自然语言。

    例：
        [{'dimension': '风格', 'answer': '写实摄影'}, {'dimension': '光影', 'answer': '黄昏'}]
        → "风格:写实摄影，光影:黄昏"
    """
    if not select_result:
        return fallback
    parts: List[str] = []
    for item in select_result:
        try:
            if isinstance(item, dict):
                dim = str(item.get("dimension") or item.get("question") or item.get("key") or "").strip()
                ans = item.get("answer") if "answer" in item else item.get("value")
                if ans is None:
                    # 兜底：取 dict 里除 dimension/question/key 之外的第一个值
                    for k, v in item.items():
                        if k not in {"dimension", "question", "key"}:
                            ans = v
                            break
                ans_text = ""
                if isinstance(ans, list):
                    try:
                        ans_text = "/".join(str(x) for x in ans if x is not None)
                    except Exception:
                        ans_text = str(ans)
                elif ans is not None:
                    ans_text = str(ans)
                if dim and ans_text:
                    parts.append(f"{dim}{item_sep}{ans_text}")
                elif ans_text:
                    parts.append(ans_text)
                elif dim:
                    parts.append(dim)
            elif item is not None:
                text = str(item).strip()
                if text:
                    parts.append(text)
        except Exception:
            continue
    if not parts:
        return fallback
    return sep.join(parts)


# ---------------- 8. Agent 工厂：默认塞好三个中间件，一行创建带监控的 agent ----------------


def create_workflow_agent(
    model,
    *,
    node_name: str,
    thread_id: str,
    user_id: str,
    system_prompt: Union[str, SystemMessage, None] = None,
    tools: Optional[Sequence[Union["BaseTool", Callable[..., Any], Dict[str, Any]]]] = None,
    extra_middleware: Optional[Sequence["AgentMiddleware"]] = None,
    # ---- 工具中间件调优参数（暴露出来，无需手动实例化 ToolMonitorMiddleware） ----
    slow_tool_threshold_sec: float = 5.0,
    retry_on_transient: int = 1,
    # ---- 剩余 create_agent 参数完整透传 ----
    response_format: Any = None,
    state_schema: Any = None,
    context_schema: Any = None,
    checkpointer: Any = None,
    store: Any = None,
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None,
    debug: bool = False,
    name: Optional[str] = None,
    cache: Optional[BaseCache[Any]] = None,
    transformers: Optional[Sequence[Any]] = None,
):
    """工作流节点专用 agent 工厂：一行创建带「LLM 监控 + 工具监控 + SSE 状态」
    三个默认中间件的 LangChain agent。

    必选位置/关键字参数（和业务上下文绑定，必须显式传）：
        model:        LLM 实例或字符串（由 create_agent 透传）
        node_name:    当前节点名（和 nodes.py 的函数名对齐即可）
        thread_id:    工作流 thread_id（通常就是 state["threadId"]）
        user_id:      用户 id

    常用可选参数：
        system_prompt:  同 create_agent
        tools:          同 create_agent（Callable / BaseTool / dict 都可以）
                        ✨ 每个 agent 节点可以传完全不同的 tools 列表（绑定专属工具）
        extra_middleware: 你自己想再额外加的中间件列表（会拼在三个默认之后）
                        ✨ 每个 agent 节点可以传完全不同的 extra_middleware（绑定专属中间件）

    ✨ 不同节点「绑定不同 tools + 不同中间件」的 3 个配方：

    (1) 只改 tools（最简单）：
        agent_A = create_workflow_agent(..., tools=[time_tool, weather_tool])
        agent_B = create_workflow_agent(..., tools=[image_safety_check_tool])

    (2) 改 tools + 加自定义中间件（最灵活）：
        agent_C = create_workflow_agent(
            ...,
            tools=[do_something_risky],
            extra_middleware=[  # 自定义的 AgentMiddleware 放在这里
                ContentGuardrailMiddleware(banned_topics=["..."]),
                CostControlMiddleware(budget_tokens=50000),
            ],
        )

    (3) 调优默认中间件的行为（不用自己实例化）：
        agent_D = create_workflow_agent(
            ...,
            tools=[upload_big_file_tool],
            slow_tool_threshold_sec=30.0,   # 工具慢日志阈值加大
            retry_on_transient=3,           # 网络失败重试次数加大
        )

    示例（test_node 里用一行搞定）：
        agent = create_workflow_agent(
            model=llm,
            node_name="test_node",
            thread_id="12332",
            user_id="55542",
            system_prompt=prompt,
            tools=[time_tool],
        )
        res = await agent.ainvoke({"messages": messages_to_langchain([...])})
    """
    # 这里用延迟 import，防止 common_node.py 被非 workflow 场景（纯解析工具函数）import 时
    # 把 agent 相关大模块也拉起来导致循环依赖 / import 变慢
    from langchain.agents import create_agent as _lc_create_agent
    from app.core.middleware import (
        LLMMonitorMiddleware,
        ToolMonitorMiddleware,
        AgentStatusMiddleware,
    )

    # 1) 默认三个中间件（顺序：abefore_agent → awrap_model_call → awrap_tool_call → aafter_agent）
    default_middleware: List["AgentMiddleware"] = [
        LLMMonitorMiddleware(
            node_name=node_name,
            thread_id=thread_id,
            user_id=user_id,
        ),
        ToolMonitorMiddleware(
            node_name=node_name,
            thread_id=thread_id,
            user_id=user_id,
            slow_tool_threshold_sec=slow_tool_threshold_sec,
            retry_on_transient=retry_on_transient,
        ),
        AgentStatusMiddleware(
            node_name=node_name,
            thread_id=thread_id,
            user_id=user_id,
        ),
    ]

    # 2) 拼 extra_middleware（用户自定义的放后面，洋葱圈里越靠后越先被包到内层——不影响一般场景）
    combined = list(default_middleware)
    if extra_middleware:
        combined.extend(extra_middleware)

    # 3) 调用官方 create_agent，把剩余关键字参数完整透传（保持 API 兼容性）
    return _lc_create_agent(
        model=model,
        tools=tools or [],
        system_prompt=system_prompt,
        middleware=combined,
        response_format=response_format,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
        transformers=transformers,
    )
