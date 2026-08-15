"""生图服务分发器：按模型厂商分发到各厂商生图服务，并统一回传保存结果。

各厂商实现各自的服务文件（配置/提示词/请求体都在各自文件内优化）：
- app.tools.image_generation.seedream → 火山引擎 Seedream（OpenAI 兼容 images.generate）
- app.tools.image_generation.qwen     → 千问 qwen-image / 万相 wan（multimodal-generation 同步接口）

本文件只做三件事：
1. resolve_image_model_name：前端友好模型名 → 厂商真实模型 ID
2. 按模型名前缀分发到对应厂商服务
3. update_work_image：把生成的图片统一保存到业务端 work 记录，并回写可访问 URL
"""
import base64
import logging
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from app.tools.common.file_download import cos_download_headers
from app.tools.image_generation.qwen import generate_qwen_images
from app.tools.image_generation.seedream import generate_seedream_images
from app.tools.image_generation.work_status import update_work_image

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_MODEL = "qwen-image-2.0-pro-2026-06-22"
MODEL_NAME_MAP = {
    "DouBao-Seedream-5.0-Pro": "doubao-seedream-5-0-pro-260628",
    "DouBao-Seedream-5.0-Lite": "doubao-seedream-5-0-260128",
    "Wan2.7": "wan2.7-image-pro",
    "qwen-image-3.0": "qwen-image-3.0",
    "qwen-image-3.0-pro": "qwen-image-3.0-pro",
    "qwen-image-2.0-pro": "qwen-image-2.0-pro-2026-06-22",
    "default": DEFAULT_IMAGE_MODEL,
}

# 参考图相关：模型侧单张图片上限（qwen-image 文档要求不超过 10MB）
MAX_REF_IMAGE_BYTES = 10 * 1024 * 1024
REF_IMAGE_DOWNLOAD_TIMEOUT = 30.0
_IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}




def resolve_image_model_name(model_name: Optional[str]) -> str:
    """模型名解析成厂商真实模型。"""
    if not model_name:
        return DEFAULT_IMAGE_MODEL
    if model_name in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[model_name]
    return DEFAULT_IMAGE_MODEL


def _normalize_images(images: Any) -> list[str]:
    """参考图统一转成 ["url1","url2"] 格式：支持 None / str / {"url": ...} / [{"url": ...}] / [str]"""
    if not images:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, dict):
        return [images["url"]] if images.get("url") else []
    result: list[str] = []
    for item in images:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and item.get("url"):
            result.append(item["url"])
    return result


async def _ref_image_to_data_uri(url: str) -> str:
    """把参考图 URL 下载成 base64 data URI，供生图模型直接使用。

    背景：用户上传的参考图存储在私有 COS（开防盗链，拒绝空 Referer），
    生图模型（qwen-image/wan/seedream）直接下载该 URL 会 403。
    因此应用侧带上白名单 Referer 自行下载，再以 data:{MIME};base64,{data}
    形式传给模型，绕开模型侧的外网下载。
    """
    if url.startswith("data:"):
        return url  # 已是 data URI，直接透传
    try:
        async with httpx.AsyncClient(
            timeout=REF_IMAGE_DOWNLOAD_TIMEOUT, follow_redirects=True
        ) as client:
            resp = await client.get(url, headers=cos_download_headers())
            resp.raise_for_status()
            body = resp.content
            content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if len(body) > MAX_REF_IMAGE_BYTES:
            raise RuntimeError(f"参考图超过 {MAX_REF_IMAGE_BYTES // (1024 * 1024)}MB，无法作为模型输入")
        if not content_type or "/" not in content_type:
            ext = urlparse(url).path.lower()
            ext = f".{ext.rsplit('.', 1)[-1]}" if "." in ext else ""
            content_type = _IMAGE_MIME_BY_EXT.get(ext, "image/jpeg")
        return f"data:{content_type};base64,{base64.b64encode(body).decode()}"
    except httpx.HTTPError as e:
        raise RuntimeError(f"参考图下载失败（可能无访问权限）: {e}") from e


async def generate_images_and_save(
    client,
    *,
    model_name: str,
    prompt: str,
    work_id: str,
    params: Optional[dict] = None,
    config: Optional[dict] = None,
    images: Any = None,
) -> tuple[list[str], int]:
    """调用生图模型并把结果保存到业务端 work 记录。

    按模型厂商分发：
    - 千问 qwen-image / 万相 wan（qwen-*/wan-*）→ qwen.py
    - 火山引擎 Seedream（doubao-* 等）→ seedream.py

    参数：
        client:      get_image_client() 返回的 AsyncOpenAI 客户端（仅 seedream 分支使用）
        model_name:  生图模型名（如 doubao-seedream-5-0-260128 / qwen-image-3.0-pro）
        prompt:      绘图提示词（只放画面内容；前缀/张数/尺寸等由各厂商服务自行拼装）
        work_id:     业务端 work 记录 id（通常是 state["threadId"]）
        params:      前端原始参数（imageCount / imageProportion / imageQuality），各厂商转成自己的配置
        config:      兼容旧调用方的额外配置，会透传给厂商服务（params 派生值优先）
        images:      参考图（图生图专用）；私有 COS 参考图会先下载转成 base64 data URI 再传给接口
                     （模型直接拉 COS URL 会被防盗链 403），以 "images" 字段传给接口

    返回：
        (最终图片URL列表, 原始生成张数)
        最终列表优先取业务端保存后的可访问 URL（preview_urls），否则用模型原始返回。
    """
    image_list = _normalize_images(images)  # 统一转化为["url1","url2"]的格式
    if image_list:
        # 私有 COS 参考图模型直接拉会 403（防盗链），先下载成 base64 data URI 再传给模型
        logger.info("参考图共 %s 张，开始下载并转 base64 供模型使用", len(image_list))
        image_list = [await _ref_image_to_data_uri(url) for url in image_list]
    if model_name.startswith("qwen") or model_name.startswith("wan"):
        # 千问/万相：参数统一由 qwen.py 内部分层组装（公共配置 + 各模型专属配置）
        image_urls = await generate_qwen_images(
            model_name,
            prompt,
            images=image_list,
            params=params or {},
        )
    else:
        # 火山引擎 Seedream
        image_urls = await generate_seedream_images(
            client,
            model_name,
            prompt,
            params=params,
            config=config,
            images=image_list,
        )

    res_count = len(image_urls)
    logger.info("模型%s生成了%d张图片", model_name, res_count)
    # 统一回传：保存到业务端 work 记录，并尽量用保存后的可访问 URL
    resp = await update_work_image(work_id, dataList=[{"url": u} for u in image_urls])
    if resp and resp.get("success"):
        saved_list = ((resp.get("data") or {}).get("dataList")) or []
        preview_urls = [item.get("url") for item in saved_list if item.get("url")]
        if preview_urls:
            image_urls = preview_urls
    return image_urls, res_count
