from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
避免：让它穿上粉色衣服。"""
# 提示词优化模板：传 prompt 时使用，一条 langChain 链完成"生成/优化"，无需 langGraph 与 SSE
# 将详细生成规则（PROMPT_RULE）内嵌，仅保留 {prompt} 作为模板变量
PROMPT_TEMPLATE = (
    "你是专业的提示词优化专家。将用户输入的描述优化为一条高质量绘图提示词，确保 AI 绘图模型能精准还原用户意图。\n\n"
    + PROMPT_RULE
    + "\n\n用户原始描述：{prompt}\n\n请直接输出优化后的提示词，不要任何解释、前缀或多余文字。"
)

# 每日灵感模板：不传 prompt 时使用，生成可直接绘图的灵感主题
INSPIRATION_TEMPLATE = """你是 AI 绘图每日灵感策划师。为用户生成一个有趣、有画面感、可直接绘图的每日灵感主题。

要求：
1. 主题要具体且可立即用于绘图，涵盖 主体 + 行为 + 环境
2. 可适当补充 风格、色彩、光影 等美学提示，让画面更出彩
3. 输出一句完整连贯的灵感主题即可

请输出今日灵感："""

# 惰性构建链，避免模块导入时的额外开销
_chain = None
_inspiration_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = (
            ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            | get_model("prompt_combined")
            | StrOutputParser()
        )
    return _chain


def _get_inspiration_chain():
    global _inspiration_chain
    if _inspiration_chain is None:
        _inspiration_chain = (
            ChatPromptTemplate.from_template(INSPIRATION_TEMPLATE)
            | get_model("prompt_combined")
            | StrOutputParser()
        )
    return _inspiration_chain


class PromptGenerateRequest(BaseModel):
    prompt: str = ""


@router.post("/generate")
async def generate_prompt(request: PromptGenerateRequest):
    """提示词生成与优化：传 prompt 优化提示词；不传则生成每日绘图灵感"""
    if request.prompt and request.prompt.strip():
        result = await _get_chain().ainvoke({"prompt": request.prompt.strip()})
        return Result.ok(data={"prompt": result, "type": "optimize"})
    try:
        result = await _get_inspiration_chain().ainvoke({})
        return Result.ok(data={"prompt": result, "type": "inspiration"})
    except Exception as e:
        return Result.fail(msg=f"每日灵感生成失败：{e}")
