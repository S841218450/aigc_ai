import logging
import re

from langgraph.types import StreamWriter
from pydantic import BaseModel, Field

from app.core.agents.model_factory import get_model, get_image_client
from app.core.prompts.prompts_factory import get_prompt
from app.tools.image_generation.work_status import update_work_image
from app.tools.image_generation.generate import generate_images_and_save, resolve_image_model_name
from app.workflows.common.common_node import clean_return, structured_output_invoke
from app.workflows.image_to_image.state import ImageToImageState

logger = logging.getLogger(__name__)


# ---------------- 1. 输入检查节点 ----------------

class ParamsFilterOutput(BaseModel):
    prompt: str = Field(description="去除敏感词与图像参数词后的纯净提示词")
    reason: str = Field(description="简要说明过滤掉了哪些词")

def _regex_filter_prompt(text: str) -> str:
    _PARAM_WORD_PATTERN = re.compile(
        r"(2k|4k|8k|1080p|高清|超清|竖屏|横屏|方形|\d+\s*张|参考强度|参考图|尺寸|比例|张数)",
        re.IGNORECASE,
    )
    return _PARAM_WORD_PATTERN.sub(" ", text or "").strip()
async def input_check_node(state: ImageToImageState):
    """输入检查节点：剔除敏感词 + 图像参数词。

    图像参数（尺寸/张数/参考强度）由 params 决定，必须在提示词里剔除，
    否则会和 generate_node 的 extra_body 参数冲突。
    """
    llm = get_model("summarizer")
    base_prompt = get_prompt("image_to_image", "params_filter")
    prompt = f"""
        {base_prompt}
        用户原始提示词：{state.get("question") or state.get("prompt", "")}
        当前已由参数控制的图像配置（以下内容不要再出现在提示词里）：
        {state.get("params") or {}}
    """
    fallback = ParamsFilterOutput(
        prompt=_regex_filter_prompt(state.get("question") or state.get("prompt", "")),
        reason="结构化解析失败，使用正则兜底过滤图像参数词",
    )
    result: ParamsFilterOutput = await structured_output_invoke(
        llm, prompt, ParamsFilterOutput, fallback,
    )

    clean_prompt = (result.prompt or "").strip() or fallback.prompt
    return clean_return({
        "agent_log": f"已完成参数过滤：{result.reason}",
        "clean_prompt": clean_prompt,
        "filter_reason": result.reason,
    })


# ---------------- 2. 提示词优化节点 ----------------

class PromptOptimizationOutput(BaseModel):
    prompt: str = Field(description="优化后的高质量绘图提示词")


async def prompt_optimize_node(state: ImageToImageState):
    """提示词优化节点：把过滤后的提示词扩写为 AI 绘图模型听得懂的高质量提示词。"""
    llm = get_model("summarizer")
    base_prompt = get_prompt("image_to_image", "prompt_optimization")
    source_prompt = state.get("clean_prompt") or state.get("question") or state.get("prompt", "")
    prompt = f"""
        {base_prompt}
        待优化提示词：{source_prompt}
    """
    fallback = PromptOptimizationOutput(prompt=source_prompt)
    result: PromptOptimizationOutput = await structured_output_invoke(
        llm, prompt, PromptOptimizationOutput, fallback,
    )

    final_prompt = (result.prompt or "").strip() or fallback.prompt
    # 回写最终提示词到业务端（非关键路径：失败只打日志，不中断出图主流程）
    try:
        await update_work_image(state["threadId"], prompt=final_prompt)
    except Exception as e:
        logger.warning(
            "回写最终提示词失败，不影响主流程：work_id=%s, prompt=%r, error=%s",
            state.get("threadId"), final_prompt[:80], e,
        )

    return clean_return({
        "agent_log": "提示词优化完成，已生成高质量绘图提示词",
        "prompt": final_prompt,
    })


# ---------------- 3. 生图节点 ----------------

async def generate_node(state: ImageToImageState, writer: StreamWriter = None):
    """图片生成节点：params 决定图像参数，prompt 只负责画面内容。

    writer 由 LangGraph 在 custom 流式模式下注入，用于模型调用前发出节点开始信号。
    """
    client = get_image_client()
    model_name = resolve_image_model_name(state.get("model", "default"))
    # 参考图：取第一张（其余仅打日志，避免多图参数格式不确定影响出图）
    origin_images = state.get("originImageList") or []
    logger.info("参考图共收到 %s 张", len(origin_images))

    reference_url = origin_images[0] if len(origin_images) == 1 else origin_images
    # 节点开始信号：模型调用前发出，SSE 先展示"正在生成图片"
    if writer:
        writer({"node": "generate_node", "messages": "正在生成图片..."})
    # 生图失败/上传失败都抛异常：交给 with_auto_retry 自动重试，仍失败则走手动重试
    try:
        image_urls, count = await generate_images_and_save(
            client,
            model_name=model_name,
            prompt=state["prompt"],
            images=reference_url,
            work_id=state["threadId"],
        )
    except Exception as e:
        logger.error("生图失败: work_id=%s, error=%s", state.get("threadId"), e)
        raise RuntimeError(f"图片生成失败：{e}")

    return clean_return({
        "agent_log": f"图片生成完成，共 {count} 张",
        "image_list": image_urls,
        "metadata": {},
    })
