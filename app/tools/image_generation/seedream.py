"""火山引擎 Seedream 生图服务：OpenAI 兼容的 images.generate 接口。

与千问/万相（multimodal-generation）不同，Seedream 走火山引擎的 OpenAI 兼容端点：
    client.images.generate(model=..., prompt=..., extra_body={...})

文生图（text_to_image）与图生图（image_to_image）共用本模块：
- text_to_image：params 带 imageCount/imageProportion/imageQuality，组图走官方 sequential 字段
- image_to_image：不传 params，提示词已由上游优化好，原样透传
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def generate_seedream_images(
    client,
    model_name: str,
    prompt: str,
    *,
    params: Optional[dict] = None,
    config: Optional[dict] = None,
    images: Any = None,
) -> list[str]:
    """调用火山引擎 Seedream 生图（OpenAI 兼容 images.generate）。

    参数：
        client:      get_image_client() 返回的 AsyncOpenAI 客户端
        model_name:  生图模型名（如 doubao-seedream-5-0-pro-260628）
        prompt:      绘图提示词（只放画面内容；前缀/张数/宽高比由本函数按 params 拼装）
        params:      前端原始参数（imageCount / imageProportion / imageQuality）
        config:      兼容旧调用方的额外配置，会合并进 extra_body（params 派生值优先）
        images:      参考图（图生图专用），以 "images" 字段传给接口

    返回：
        模型生成的图片 URL 列表（原始返回，未转存业务端）。
    """
    # 从前端 params 派生张数/尺寸（兼容旧 config 的组图 max_images / size）
    image_count = 1
    image_quality = None
    if params:
        image_count = int(params.get("imageCount") or 1)
        image_quality = params.get("imageQuality")
    elif config:
        opts = config.get("sequential_image_generation_options") or {}
        image_count = int(opts.get("max_images") or 1)
        image_quality = config.get("size")

    extra_body = {
        **(config or {}),
        "response_format": "url",
        "watermark": False,
    }
    if image_quality:
        extra_body["size"] = image_quality
    if images is not None:
        # generate.py 已统一转成 ["url1","url2"]，直接透传
        extra_body["images"] = list(images)

    if params:
        # text_to_image：按单图/组图拼中文前缀，明确张数与宽高比
        if image_count > 1:
            extra_body["sequential_image_generation"] = "auto"
            extra_body["sequential_image_generation_options"] = {"max_images": image_count}
            image_proportion = params.get("imageProportion", "1:1")
            final_prompt = f"生成一组一共{image_count}张宽高比为{image_proportion}的图片,图片要求为{prompt}"
        else:
            final_prompt = f"生成一张图片,图片要求为{prompt}"
    else:
        # image_to_image 等旧调用：提示词已由上游优化好，原样透传
        final_prompt = prompt

    response = await client.images.generate(model=model_name, prompt=final_prompt, extra_body=extra_body)
    res_list = response.data or []
    image_urls = [item.url for item in res_list if item.url]
    logger.info("Seedream生图完成 model=%s 生成%d张", model_name, len(image_urls))
    return image_urls
