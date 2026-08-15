from typing import Optional
import json
import re

from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from app.core.agents.model_factory import get_model
from app.models.schemas.common import Result

router = APIRouter()

PROMPT_RULE = """
提示词生成规则：
1.用自然语言清晰描述画面
建议用简洁连贯的自然语言写明 主体 + 行为 + 环境，若对画面美学有要求，可用自然语言或短语补充 风格、色彩、光影、构图 等美学元素。
示例：一个穿着华丽服装的女孩，撑着遮阳伞走在林荫道上，莫奈油画风格。
避免：一个女孩，撑伞，林荫街道，油画般的细腻笔触。
2.明确应用场景和用途
当有明确的应用场景时，推荐在文本提示中写明图像用途和类型。
示例：设计一个游戏公司的 logo，主体是一只在用游戏手柄打游戏的狗，logo 上写有公司名 “PITBULL”。
避免：一张抽象图片，狗拿着游戏手柄，狗狗上写 PITBULL。
3.提升风格渲染效果
如果有明确的风格需求，使用精准的 风格词 或提供 参考图像，能获得更理想的效果。
4.提高文本渲染准确度
建议将要生成的 文字内容 放在 双引号 中。
示例：生成一张海报，标题为 “Seedream 4.5”
避免：生成一张海报，标题为 Seedream 4.5
5.明确图片编辑目标和希望保持不变的部分
使用 简洁明确的指令，说明需要修改或参考的对象及具体操作，避免使用指代模糊的代词；如果希望除了修改的内容都保持不变，则可以在 prompt 中强调。
示例：让图中最高的那只熊猫穿上粉色的京剧服饰并戴上头饰，并保持动作不变。
避免：让它穿上粉色衣服。
"""


def get_prompt(prompt: Optional[str], style: Optional[str] = None) -> str:
    """按场景拼接系统提示词：传 prompt 走提示词优化，否则走每日灵感生成"""
    base_rule = PROMPT_RULE  # 提示词生成规则
    str_output = """
请严格输出 JSON 格式，不要输出任何 JSON 以外的文字：
{
    "prompt": "生成的提示词"
}
"""
    if prompt:
        # 提示词优化
        base_prompt = f"""
你是一名提示词优化专家，能根据用户给出的提示词进行优化。

用户给出的提示词：{prompt}

要求：
1. 根据提示词生成规则生成提示词
"""
    else:
        # 每日灵感生成
        base_prompt = f"""
你是 AI 绘图灵感的策划师。围绕给定的风格类别，构思一个足够新奇、有画面感的提示词。
可以从以下几个元素出发：
1. 核心主题（core_theme）：画面主体、场景、事件
2. 色彩体系（color_system）：主色调、冷暖、饱和度等
3. 光影氛围（light_atmosphere）：光源、明暗、情绪氛围
4. 构图镜头（frame_composition）：视角、画幅、构图方式
5. 主体细节（object_detail）：材质、纹理、状态、特征
6. 背景环境（environment_bg）：背景类型、环境元素

用户给出的图片风格：{style}

要求：
1. 根据用户给出的风格进行提示词生成，如果是"智能匹配"则允许自主发挥，否则需要仅仅围绕风格生成
2. 主体要具体、独特、可绘制，避免"一只猫""一片森林"这类烂大街的题材
3. 大胆混搭、反转常识，优先选不容易撞车的组合
4. 关键词给 3-5 个最能落地的画面元素
5. 根据提示词生成规则生成提示词
"""
    return f"{base_prompt}\n{base_rule}\n{str_output}"


class PromptGenerateRequest(BaseModel):
    prompt: Optional[str] = ""
    style: Optional[str] = "智能匹配"


def _parse_prompt_result(text: str) -> str:
    """解析模型输出的 {"prompt": "..."}，兼容 markdown 代码块包裹"""
    text = (text or "").strip()
    # 去掉 ```json ... ``` 围栏
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        prompt_text = data.get("prompt") if isinstance(data, dict) else None
        if prompt_text:
            return prompt_text
    except Exception:
        pass
    # 兜底：正则提取 "prompt" 字段
    m = re.search(r'"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        return m.group(1).replace('\\"', '"')
    raise ValueError(f"模型输出解析失败：{text[:200]}")


async def generate_prompt(prompt: Optional[str], style: Optional[str] = None) -> str:
    """生成/优化提示词：create_agent 按系统提示词要求输出 JSON，解析出最终 prompt 文本"""
    agent = create_agent(
        model=get_model("prompt_combined", {"temperature": 0.9}),
        system_prompt=get_prompt(prompt, style),
        tools=[],
    )
    res = await agent.ainvoke({
        "messages": [HumanMessage(content="请按要求生成提示词。")]
    })
    content = res["messages"][-1].content
    if isinstance(content, list):
        # 兼容多模态返回（content 为内容块列表）
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return _parse_prompt_result(content)


@router.post("/generate")
async def generate_prompt_api(request: PromptGenerateRequest):
    """提示词生成与优化：传 prompt 优化提示词；不传则生成每日绘图灵感；style 控制画面风格"""
    prompt = (request.prompt or "").strip() or None
    style = (request.style or "").strip() or None
    try:
        result = await generate_prompt(prompt, style)
        return Result.ok(
            data={"prompt": result, "type": "optimize" if prompt else "inspiration"}
        )
    except Exception as e:
        return Result.fail(msg=f"提示词生成失败：{e}")
