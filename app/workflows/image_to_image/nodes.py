import asyncio
import functools
import logging
import re
from typing import List, Optional

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.core.agents.model_factory import get_model, get_image_client
from app.core.prompts.prompts_factory import get_prompt
from app.tools.image_generation.work_status import upload_file_by_url, update_work_image
from app.utils.image_utils import SIZE_MAP
from app.workflows.common.common_node import clean_return, structured_output_invoke
from app.workflows.image_to_image.state import ImageToImageState

logger = logging.getLogger(__name__)

# ---------------- 节点重试机制 ----------------

MAX_AUTO_RETRIES = 2     # 节点自动重试次数（指数退避）
MAX_MANUAL_RETRIES = 3   # 手动重试总轮数上限（防死循环）


def with_auto_retry(node_fn, max_retries: int = MAX_AUTO_RETRIES, base_delay: float = 1.0):
    """节点自动重试装饰器：节点抛异常时自动重试，仍失败则写入 node_error/retry_target，
    由图路由到 await_retry_node 等待用户手动重试（/retry 端点用 Command(resume=True) 恢复）。
    """
    @functools.wraps(node_fn)
    async def wrapper(state):
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = await node_fn(state)
                if result is None:
                    result = {}
                # 成功后清除上次失败标记，让路由走正常后续节点
                result["node_error"] = None
                result["retry_target"] = None
                return result
            except Exception as e:
                last_error = e
                logger.warning("[%s] 第 %s 次执行失败: %s", node_fn.__name__, attempt + 1, e)
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** attempt))
        return clean_return({
            "agent_log": f"{node_fn.__name__} 执行失败（自动重试 {max_retries} 次仍未成功），等待手动重试",
            "node_error": str(last_error),
            "retry_target": node_fn.__name__,
        })

    return wrapper


async def await_retry_node(state: ImageToImageState):
    """手动重试中断门：节点失败后暂停，等用户触发重试后回到失败节点继续执行。"""
    retry_count = (state.get("retry_count") or 0) + 1
    interrupt({
        "title": "节点执行失败，等待重试",
        "message": f"节点「{state.get('retry_target')}」执行失败，点击「重试」后将从该节点继续执行",
        "retry_target": state.get("retry_target"),
        "retry_count": retry_count,
        "node_error": state.get("node_error"),
    })
    return clean_return({
        "agent_log": f"用户触发第 {retry_count} 轮手动重试，回到节点「{state.get('retry_target')}」",
        "retry_count": retry_count,
    })


# ---------------- 1. 参数过滤节点 ----------------

class ParamsFilterOutput(BaseModel):
    prompt: str = Field(description="去除敏感词与图像参数词后的纯净提示词")
    reason: str = Field(description="简要说明过滤掉了哪些词")

def _regex_filter_prompt(text: str) -> str:
    _PARAM_WORD_PATTERN = re.compile(
        r"(2k|4k|8k|1080p|高清|超清|竖屏|横屏|方形|\d+\s*张|参考强度|参考图|尺寸|比例|张数)",
        re.IGNORECASE,
    )
    return _PARAM_WORD_PATTERN.sub(" ", text or "").strip()
async def params_filter_node(state: ImageToImageState):
    """参数过滤节点：剔除敏感词 + 图像参数词。

    图像参数（尺寸/张数/参考强度）由 params 决定，必须在提示词里剔除，
    否则会和 generate_image_node 的 extra_body 参数冲突。
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


async def prompt_optimization_node(state: ImageToImageState):
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


async def generate_image_node(state: ImageToImageState):
    """图片生成节点：params 决定图像参数，prompt 只负责画面内容。"""
    client = get_image_client()
    model_name = "doubao-seedream-5-0-260128"
    # 参考图：取第一张（其余仅打日志，避免多图参数格式不确定影响出图）
    origin_images = state.get("originImageList") or []
    logger.info("参考图共收到 %s 张", len(origin_images))
    
    reference_url = origin_images[0] if len(origin_images) ==1 else origin_images
    # 生图失败/上传失败都抛异常：交给 with_auto_retry 自动重试，仍失败则走手动重试
    try:
        response = await client.images.generate(
            model=model_name,
            prompt=state["prompt"],
            extra_body={
                "images": reference_url,
                "response_format": "url",
                "watermark": False,#水印
            },
        )
    except Exception as e:
        logger.error("生图失败: work_id=%s, error=%s", state.get("threadId"), e)
        raise RuntimeError(f"图片生成失败：{e}")
    res_list = response.data
    image_urls = [item.url for item in res_list if item.url]
    logger.info(f"模型{getattr(response, 'model', model_name)}生成了{len(res_list)}张图片")
    resp = await update_work_image(state["threadId"], dataList=[{"url": u} for u in image_urls])
    if resp and resp.get("success"):
        saved_list = ((resp.get("data") or {}).get("dataList")) or []
        preview_urls = [item.get("url") for item in saved_list if item.get("url")]
        if preview_urls:
            image_urls = preview_urls

    return clean_return({
        "agent_log": f"图片生成完成，共 {len(res_list)} 张，正在评估质量...",
        "image_list": image_urls,
        "metadata": {},
    })


# ---------------- 4. 质量评估节点 ----------------

class QualityOutput(BaseModel):
    isPass: bool = Field(description="图片质量是否合格")
    match_score: int = Field(description="图片与提示词匹配度 0-10")
    image_problem: str = Field(description="当前图片存在的缺陷问题")
    modify_suggest: str = Field(description="可直接复用的绘图优化建议")


async def quality_evaluation_node(state: ImageToImageState):
    """生图质量审核与评估：评估生成结果是否合格，仅做记录，不阻塞主流程。"""
    image_list = state.get("image_list") or []
    if not image_list:
        return clean_return({
            "agent_log": "本次未生成有效图片，跳过质量评估",
            "isPass": False,
        })

    llm = get_model("summarizer")
    base_prompt = get_prompt("image_to_image", "quality_evaluation")
    prompt = f"""
        {base_prompt}
        用户提示词：{state.get("question") or state.get("prompt", "")}
        生成图片URL：{image_list[0]}
    """
    fallback = QualityOutput(
        isPass=True,
        match_score=7,
        image_problem="结构化解析失败，无法判断具体缺陷",
        modify_suggest="",
    )
    result: QualityOutput = await structured_output_invoke(
        llm, prompt, QualityOutput, fallback,
    )

    return clean_return({
        "agent_log": f"图片评估完成，匹配度：{result.match_score}/10，问题：{result.image_problem}",
        "isPass": result.isPass,
        "match_score": result.match_score,
        "image_problem": result.image_problem,
    })


# ---------------- 5. 总结节点 ----------------

async def summary_node(state: ImageToImageState):
    """总结节点：汇总本次生图结果，返回最终图片列表与说明。"""
    image_list = state.get("image_list") or []
    is_pass = state.get("isPass", True)
    image_problem = state.get("image_problem") or ""

    if image_list:
        if is_pass:
            answer = f"已成功生成 {len(image_list)} 张图片。"
        else:
            answer = f"已生成 {len(image_list)} 张图片，但质量评估未完全通过：{image_problem}"
    else:
        answer = "本次图片生成失败，请检查输入后重试。"

    return clean_return({
        "agent_log": "图生图流程执行完成",
        "answer": answer,
    })
