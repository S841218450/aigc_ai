import logging
from typing import Dict, List

from langgraph.types import interrupt, StreamWriter
from pydantic import BaseModel, Field

from app.core.agents.model_factory import get_model, get_image_client
from app.core.prompts.prompts_factory import get_prompt

from app.tools.image_generation.work_status import update_work_image
from app.tools.image_generation.generate import generate_images_and_save, resolve_image_model_name
from app.workflows.common.common_node import (
    clean_return,
    format_select_result,
    structured_output_invoke,
)
from app.workflows.text_to_image.state import TextToImageState


logger = logging.getLogger(__name__)


class JudgeOutput(BaseModel):
    totalScope: int = Field(description="描述总分 0-70")
    need_manual_count: int = Field(description="薄弱维度数量，决策路由核心参数")
    judgeList: Dict[str, int] = Field(description="各维度名称+对应分数明细")
    judge_summary: str = Field(description="针对用户描述的具体缺陷总结")


async def input_check_node(state: TextToImageState, writer: StreamWriter):
    """输入检查节点：对用户描述进行评分评估，输出决策所需分数与薄弱维度。"""
    llm = get_model('intent')
    user_question = state["question"]
    prompt_template = get_prompt('text_to_image', 'descScopeJudge')
    prompt = prompt_template.format(question=user_question, params=state["params"])

    writer({"node": "input_check_node"})
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


async def decision_node(state: TextToImageState, writer: StreamWriter):
    """方案决策节点：意图+分数双层判定，输出是否放行生图（不通过则进入补充描述）。"""
    llm = get_model('Supervisor')
    prompt_template = get_prompt('text_to_image', 'decisionRouter')
    prompt = prompt_template.format(
        question=state["question"],
        params=state["params"],
        totalScope=state["totalScope"],
        judgeList=state["judgeList"],
        need_manual_count=state["need_manual_count"],
    )

    writer({"node": "decision_node"})
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
    """补充描述节点：根据薄弱维度生成选择项给用户补全描述。"""
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


async def interrupt_node(state: TextToImageState):
    """补充描述中断节点：暂停等待用户选择题结果，返回后按结果决定是否继续补充。"""
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


class PromptOptimizeOutput(BaseModel):
    prompt: str = Field(description="最终提示词")


async def prompt_optimize_node(state: TextToImageState, writer: StreamWriter):
    """提示词优化节点：合并补充选择结果，优化生成最终的绘图提示词。"""
    llm = get_model('prompt_combined')
    prompt_template = get_prompt('common', 'prompt_combined')
    prompt = prompt_template.format(
        question=state["question"],
        params=state["params"],
        selectResult=state["selectResult"],
    )

    writer({"node": "prompt_optimize_node"})
    # 兜底：直接拼接 问题 + 选择结果（复用 format_select_result 统一格式）
    select_text = format_select_result(state.get("selectResult") or [], sep="，", item_sep=":", fallback="")
    fallback_prompt = (
        state["question"] + ("，" + select_text if select_text else "")
    )
    fallback = PromptOptimizeOutput(prompt=fallback_prompt)
    parse_result: PromptOptimizeOutput = await structured_output_invoke(
        llm, prompt, PromptOptimizeOutput, fallback,
    )
    final_prompt = (parse_result.prompt or "").strip() or fallback_prompt

    # 保存最终提示词（非关键路径：失败只打日志，不影响出图主流程）
    try:
        await update_work_image(state["threadId"], prompt=final_prompt)
    except Exception as e:
        logger.warning(
            "回写最终提示词失败，不影响主流程：work_id=%s, prompt=%r, error=%s",
            state.get("threadId"), final_prompt[:80], e,
        )

    return clean_return({
        "agent_log": "对用户补充描述进行合并优化，生成了最终绘图提示词",
        "prompt": final_prompt,
    })


async def generate_node(state: TextToImageState, writer: StreamWriter):
    """图片生成节点：按 model 选择生图厂商（火山引擎 Seedream / 千问 qwen-image），
    并把生成的图片保存到业务端 work 记录，回传统一为图片 URL 数组。"""
    client = get_image_client()
    # 模型选择与提示词优化
    model_name = state.get("model", "default")
    resolved_model = resolve_image_model_name(model_name)
    combined_prompt = (state.get("prompt") or "").strip()
    gen_prompt = combined_prompt or (state.get("question") or "").strip()

    # 各厂商的提示词前缀/尺寸/组图等配置由 generate_images_and_save 内部按厂商拼装
    logger.info("[生图] 模型=%s 提示词=%s params=%s", resolved_model, gen_prompt[:150], state["params"])
    # 节点开始信号：模型调用前发出，SSE 先展示"正在生成图片"
    writer({
        "node": "generate_node",
        "messages": "正在生成图片...",
    })

    image_urls, _ = await generate_images_and_save(
        client,
        model_name=resolved_model,
        prompt=gen_prompt,
        params=state["params"],
        work_id=state["threadId"],
    )
    return clean_return({
        "agent_log": f"图片生成完成,共 {len(image_urls)} 张图片",
        "image_list": image_urls,
        "metadata": {},
    })
