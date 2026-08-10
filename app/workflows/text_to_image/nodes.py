import asyncio
import logging
from datetime import datetime
from itertools import count

from typing import List, Dict, Any, Optional, Sequence
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import interrupt, StreamWriter
from pydantic import BaseModel, Field

from app.core.agents.model_factory import get_model, get_image_client
from app.core.middleware import LLMMonitorMiddleware, ToolMonitorMiddleware
from app.core.prompts.prompts_factory import get_prompt
from app.models.schemas.text_to_image import paramsType

from app.tools.image_generation.work_status import update_work_image
from app.workflows.common.common_node import (
    structured_output_invoke,
    messages_to_langchain,
    parse_agent_result,
    clean_return,
    format_select_result,
    create_workflow_agent,
    run_structured_node,
)
from app.workflows.text_to_image.state import TextToImageState



logger = logging.getLogger(__name__)


class JudgeOutput(BaseModel):
    totalScope: int = Field(description="描述总分 0-70")
    need_manual_count: int = Field(description="薄弱维度数量，决策路由核心参数")
    judgeList: Dict[str, int] = Field(description="各维度名称+对应分数明细")
    judge_summary: str = Field(description="针对用户描述的具体缺陷总结")


async def desc_code_judge_node(state: TextToImageState, writer: StreamWriter):
    """提示词描述识别评价结点"""
    llm = get_model('intent')
    user_question = state["question"]
    prompt_template = get_prompt('text_to_image', 'descScopeJudge')
    prompt = prompt_template.format(question=user_question, params=state["params"])

    writer({"node": "desc_code_judge_node"})
    fallback = JudgeOutput(
        totalScope=35,
        need_manual_count=3,
        judgeList={"构图": 5, "主体": 5, "风格": 5, "光影": 5, "色彩": 5, "细节": 5, "氛围": 5},
        judge_summary="结构化解析失败，使用默认评估（中等描述质量，建议补充选择题）",
    )
    result: JudgeOutput = await structured_output_invoke(llm, prompt, JudgeOutput, fallback)

    return clean_return({
        "agent_log": (
            f"对用户描述进行了评估，评估结果：{result.judge_summary}，"
            f"总分：{result.totalScope}，薄弱维度：{result.need_manual_count}个"
        ),
        "totalScope": result.totalScope,
        "need_manual_count": result.need_manual_count,
        "judgeList": result.judgeList,
        "judge_summary": result.judge_summary,
    })


class DecisionOutput(BaseModel):
    isPass: bool = Field(description="是否直接生图，仅允许 True / False")
    decide_result: str = Field(description="一句话决策原因")


async def decision_router(state: TextToImageState, writer: StreamWriter):
    """决策路由节点：意图+分数双层判定，输出是否放行生图"""
    llm = get_model('Supervisor')
    prompt_template = get_prompt('text_to_image', 'decisionRouter')
    prompt = prompt_template.format(
        question=state["question"],
        params=state["params"],
        totalScope=state["totalScope"],
        judgeList=state["judgeList"],
        need_manual_count=state["need_manual_count"],
    )

    writer({"node": "decision_router"})
    # 兜底策略：分数 >= 49（满分70的70%）且薄弱维度 <=1 自动放行，否则进入补充
    is_pass_default = bool(state.get("totalScope", 0) >= 49 and state.get("need_manual_count", 99) <= 1)
    fallback = DecisionOutput(
        isPass=is_pass_default,
        decide_result=(
            "结构化解析失败，使用分数兜底："
            + ("达到及格线，直接生图" if is_pass_default else "描述质量不足，需要补充")
        ),
    )
    result: DecisionOutput = await structured_output_invoke(llm, prompt, DecisionOutput, fallback)

    action = "直接生成图片" if result.isPass else "让用户补充描述"
    return clean_return({
        "agent_log": f"决策结论：{result.decide_result}，决定{action}",
        "isPass": result.isPass,
        "decide_result": result.decide_result,
    })


class SelectItem(BaseModel):
    dimension: str = Field(description="对应维度名称")
    question: str = Field(description="面向用户的提问短句")
    select_type: str = Field(description="单选/多选")
    options: List[str] = Field(description="选项列表")


class SupplementaryOutput(BaseModel):
    selectList: List[SelectItem] = Field(description="细分补充选择题列表，前端展示用")


async def supplementary_node(state: TextToImageState, writer: StreamWriter):
    """辅助描述选择题生成结点：根据薄弱维度生成选择项给用户补全描述"""
    llm = get_model('supplementary')
    prompt_template = get_prompt('text_to_image', 'optionGenerate')
    prompt = prompt_template.format(
        question=state["question"],
        params=state["params"],
        judgeList=state["judgeList"],
        need_manual_count=state["need_manual_count"],
        judge_summary=state["judge_summary"],
    )

    writer({"node": "supplementary_node"})
    # 兜底：至少生成 3 道通用选择题，保证 UI 不会空列表
    fallback = SupplementaryOutput(selectList=[
        SelectItem(dimension="风格", question="希望采用什么绘画风格？", select_type="单选",
                   options=["写实摄影", "动漫插画", "油画厚涂", "赛博朋克"]),
        SelectItem(dimension="光影", question="整体光线氛围？", select_type="单选",
                   options=["明亮日间", "黄昏夕阳", "夜晚霓虹", "戏剧化对比光"]),
        SelectItem(dimension="构图", question="构图视角？", select_type="单选",
                   options=["全景远景", "中景叙事", "特写聚焦", "俯视/仰视"]),
    ])
    parse_result: SupplementaryOutput = await structured_output_invoke(
        llm, prompt, SupplementaryOutput, fallback,
    )
    user_selectList = [item.model_dump() for item in parse_result.selectList] or [
        item.model_dump() for item in fallback.selectList
    ]

    return clean_return({
        "agent_log": f"生成了{len(user_selectList)}道选择题让用户补全描述",
        "selectList": user_selectList,
        "selectResult": None,
    })


async def human_interrupt_node(state: TextToImageState):
    """人工介入路由节点：用户选择补充描述后，根据选择结果判断是否放行生图"""
    selectResult = interrupt({
        "title": "请完善绘图描述缺失信息",
        "question_list": state["selectList"],
    })
    if selectResult:
        return clean_return({
            "agent_log": "用户选择了补充描述，继续生成图片",
            "selectResult": selectResult,
        })
    return clean_return({
        "agent_log": "用户没有选择补充描述，重新生成选择题",
        "selectResult": None,
    })


class PromptCombinedOutput(BaseModel):
    prompt: str = Field(description="最终提示词")


async def prompt_combined_node(state: TextToImageState, writer: StreamWriter):
    """提示词合并结点：优化合并生成最终的prompt"""
    llm = get_model('prompt_combined')
    prompt_template = get_prompt('common', 'prompt_combined')
    prompt = prompt_template.format(
        question=state["question"],
        params=state["params"],
        selectResult=state["selectResult"],
    )

    writer({"node": "prompt_combined_node"})
    # 兜底：直接拼接 问题 + 选择结果（复用 format_select_result 统一格式）
    select_text = format_select_result(state.get("selectResult") or [], sep="，", item_sep=":", fallback="")
    fallback_prompt = (
        state["question"] + ("，" + select_text if select_text else "")
    )
    fallback = PromptCombinedOutput(prompt=fallback_prompt)
    parse_result: PromptCombinedOutput = await structured_output_invoke(
        llm, prompt, PromptCombinedOutput, fallback,
    )
    final_prompt = (parse_result.prompt or "").strip() or fallback_prompt

    #保存最终提示词
    await update_work_image(state["threadId"], prompt=final_prompt)

    return clean_return({
        "agent_log": "对用户补充描述进行合并优化，生成了最终绘图提示词",
        "prompt": final_prompt,
    })

def get_parma_prompt(params: dict, prompt: str) -> str:
    imageCount = params.get("imageCount",1)
    quality = params.get("imageQuality","2k")
    proportion = params.get("imageProportion","1:1")
    params_prompt = f"""生成{imageCount}张{quality}宽高比为{proportion}的图片"""
    return f"""{params_prompt},图片要求为{prompt}"""

async def generate_image_node(state: TextToImageState, writer: StreamWriter):
    """图片生成结点：调用火山引擎 Seedream 模型绘图，并把生成的图片保存到业务端 work 记录"""
    client = get_image_client()
    # 模型名称映射（集中管理，后续改为 settings 配置）
    MODEL_NAME_MAP = {
        "DouBao-Seedream-5.0-Pro": "doubao-seedream-5-0-pro-260628",
    }
    DEFAULT_MODEL_NAME = "doubao-seedream-5-0-260128"

    # 断点续传：如果已有原始图片URL，跳过生图直接返回
    raw_urls = state.get("raw_image_urls")
    if not raw_urls:
        params: paramsType = state["params"]
        prompt = state["prompt"]

        model = state.get("model", "default")
        model_name = MODEL_NAME_MAP.get(model, DEFAULT_MODEL_NAME)

        # 节点开始信号：模型调用前发出，SSE 先展示"正在生成图片"
        writer({"node": "generate_image_node"})

        # AsyncOpenAI 原生异步调用，不阻塞事件循环
        response = await client.images.generate(
            model=model_name,
            prompt=get_parma_prompt(params, prompt),
            extra_body={
                "response_format": "url",
                "watermark": False,
            },
        )
        res_list = response.data or []

        image_urls = [item.url for item in res_list if item.url]
        logger.info(f"模型{getattr(response, 'model', model_name)}生成了{len(res_list)}张图片")

        resp = await update_work_image(state["threadId"], dataList=[{"url": u} for u in image_urls])
        if resp and resp.get("success"):
            saved_list = ((resp.get("data") or {}).get("dataList")) or []
            preview_urls = [item.get("url") for item in saved_list if item.get("url")]
            if preview_urls:
                image_urls = preview_urls
    else:
        # 旧 checkpoint 恢复：raw_image_urls 为字符串 URL 列表
        image_urls = raw_urls
    return clean_return({
        "agent_log": "图片生成完成，正在评估质量...",
        "image_url": image_urls,
        "metadata": {},
    })


class SummaryOutput(BaseModel):
    match_score: int = Field(description="图片和用户描述匹配度 0-10")
    image_problem: str = Field(description="当前图片存在的缺陷问题")
    modify_suggest: str = Field(description="可直接复用的绘图优化建议")
    judge_note: str = Field(description="评估判定备注")


async def summer_node(state: TextToImageState, writer: StreamWriter):
    """出图后总结评估：识别图片缺陷、返回修改建议（不再触发重绘回流）。

    用 run_structured_node 工厂把 45 行 boilerplate 收敛到 25 行左右，
    同时暴露：
    - tags: 每个节点可打自己的 tracing tag
    - callbacks: 每个节点可挂自己的 LLM 回调（相当于"不同中间件"）
    """
    llm = get_model('summarizer')

    # 兜底 —— 默认中等分数，避免影响用户正常出图结果
    fallback = SummaryOutput(
        match_score=7,
        image_problem="结构化解析失败，无法判断具体缺陷（建议人工复核）",
        modify_suggest="",
        judge_note="解析失败兜底：仅返回评估结果，不触发重绘",
    )

    return await run_structured_node(
        node_name="summer_node",
        state=state,
        llm=llm,
        output_cls=SummaryOutput,
        fallback=fallback,
        writer=writer,
        get_prompt=lambda s: get_prompt('text_to_image', 'imageSummaryJudge').format(
            question=s["question"],
            prompt=s["prompt"],
            image_url=(s.get("image_url") or [""])[0],
        ),
        # ✨ 不同节点打不同 tracing 标签
        tags=["summary"],
        # ✨ 不同节点挂不同 LLM 回调（比如 summer_node 要做"LLM 输出违规词拦截"就挂这里）
        # callbacks=[NSFWContentFilterCallback(), PromptInjectionBlockCallback()],
        callbacks=None,
        make_agent_log=lambda r, s: (
            f"图片评估完成，匹配度：{r.match_score}/10，建议：{r.modify_suggest}"
        ),
    )


def time_tool():
    """获取当前日期和时间"""
    return datetime.now()


class test_out_put(BaseModel):
    agent_log: str
    answer: str


async def test_node():
    llm = get_model('summarizer')
    prompt = (
        "你是一个生活小助手，专门回答用户的生活问题。"
        "请严格按照以下输出结构输出 Json 格式，不要输出任何额外文字：\n"
        '{"agent_log":"思考过程","answer":"回答内容"}'
    )

    # 一行创建：三个默认中间件（LLM监控 + 工具监控 + SSE状态）自动注入
    agent = create_workflow_agent(
        model=llm,
        node_name="test_node",
        thread_id="12332",
        user_id="55542",
        system_prompt=prompt,
        tools=[time_tool],
    )

    history_messages = [{"role": "user", "content": "你好，现在多少点了"}]
    fallback = test_out_put(
        agent_log="解析失败兜底：未能识别模型结构化输出",
        answer="抱歉，回答生成失败，请稍后再试。",
    )

    res = await agent.ainvoke({"messages": messages_to_langchain(history_messages)})
    parsed: test_out_put = parse_agent_result(res, test_out_put, fallback)

    print("===== 解析结果（结构化对象）=====")
    print(f"agent_log: {parsed.agent_log}")
    print(f"answer:    {parsed.answer}")


if __name__ == "__main__":
    # 传入：系统提示、历史对话、用户当前问题
    asyncio.run(test_node())


# ============================================================================
# 🌰 示例：不同节点「绑定不同 tools + 不同中间件」的完整写法
# ============================================================================
#
# 以上 summer_node 用的是 run_structured_node（纯 LLM 回调/标签差异化）。
# 下面给两个 *Agent 节点级* 的真实可运行示例，对应原写法 test_node 那种
# create_agent + tools + middleware 的自由度：
#
#   A) test_node_with_custom_tools()  —— 绑定专属 tools 列表
#   B) test_node_with_extra_mw()      —— 绑定专属 extra_middleware + 调优默认中间件参数
#
# 在实际 workflow 里写业务节点时，把这些示例里的"创建 agent 部分"搬到你的
# async def my_agent_node(state): 里即可，thread_id/user_id 直接从 state 取。
# ============================================================================


def lucky_number_tool() -> int:
    """生成一个 1-99 的随机幸运数字（作为"专属工具"示例）"""
    import random as _r
    return _r.randint(1, 99)


async def test_node_with_custom_tools():
    """🌰 示例A：绑定「专属 tools 列表」（每个节点 tools 完全不同）。"""
    llm = get_model('summarizer')
    prompt = (
        "你是一个数字占卜小助手，回答用户问题时可以调用 lucky_number_tool 取幸运数字。"
        "严格输出 JSON：{\"agent_log\":\"...\",\"answer\":\"...\"}。"
    )

    # 🔑 这里和 test_node 的区别只在于 tools 参数不一样：只绑定 time_tool？还是同时绑
    # lucky_number_tool？create_workflow_agent 每次可以传完全独立的 tools 列表，
    # 不会影响别的节点。
    agent = create_workflow_agent(
        model=llm,
        node_name="lucky_agent",
        thread_id="custom_tools_thread",
        user_id="55542",
        system_prompt=prompt,
        tools=[time_tool, lucky_number_tool],   # ✨ 绑定不同的 tools
    )

    history = [{"role": "user", "content": "给我一个幸运数字，并告诉我现在几点钟"}]
    fallback = test_out_put(
        agent_log="解析失败兜底",
        answer="抱歉，生成失败。",
    )
    res = await agent.ainvoke({"messages": messages_to_langchain(history)})
    parsed: test_out_put = parse_agent_result(res, test_out_put, fallback)
    print("===== 自定义tools示例 =====")
    print(f"agent_log: {parsed.agent_log}")
    print(f"answer:    {parsed.answer}")


# （可选）如果你还想加"专属 AgentMiddleware"，可以直接继承 AgentMiddleware。
# 这里给一个最小示例——LLM 调用成功后偷偷在 extra 里上报 trace_id：
class _DemoTraceMiddleware:
    """演示用自定义中间件（最小骨架，按需实现 abefore_agent / awrap_model_call 等钩子）。"""

    async def abefore_agent(self, state, runtime):  # type: ignore[override]
        return None

    async def aafter_agent(self, state, runtime):  # type: ignore[override]
        return None


async def test_node_with_extra_mw():
    """🌰 示例B：「专属 extra_middleware + 调优默认中间件参数」。"""
    llm = get_model('summarizer')
    prompt = (
        "你是生活小助手。严格输出 JSON：{\"agent_log\":\"...\",\"answer\":\"...\"}。"
    )

    agent = create_workflow_agent(
        model=llm,
        node_name="custom_mw_agent",
        thread_id="custom_mw_thread",
        user_id="55542",
        system_prompt=prompt,
        tools=[time_tool],
        # ✨ 配方(2): 额外追加自己写的 AgentMiddleware
        extra_middleware=[_DemoTraceMiddleware()],
        # ✨ 配方(3): 直接调优默认中间件的行为，不用手动实例化
        slow_tool_threshold_sec=1.0,   # 工具超过 1s 就打印慢预警
        retry_on_transient=2,          # 网络失败重试 2 次
    )

    history = [{"role": "user", "content": "你好，现在几点？"}]
    fallback = test_out_put(agent_log="fallback", answer="生成失败")
    res = await agent.ainvoke({"messages": messages_to_langchain(history)})
    parsed = parse_agent_result(res, test_out_put, fallback)
    print("===== 自定义中间件示例 =====")
    print(f"agent_log: {parsed.agent_log}")
    print(f"answer:    {parsed.answer}")
