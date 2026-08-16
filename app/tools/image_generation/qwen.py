"""千问（DashScope）/ 万相（wan）生图服务：multimodal-generation 接口。

统一流程：
1. 组装参数：公共配置（n/watermark/negative_prompt）+ 各模型专属配置（各模型独立分辨率映射）
2. 组装请求体：参考图已在 generate.py 统一转成 ["url1","url2"]，直接按官方 curl 格式赋值
3. 提交：
   - wan：X-DashScope-Async: enable 异步提交，拿 task_id 后指数退避轮询
   - qwen-image 系列：当前 Key 不支持异步，同步等结果（output.results[].url / choices[].message.content[].image）
4. 指数退避轮询 {host}/api/v1/tasks/{task_id}，SUCCEEDED 后取 output.choices[].message.content[].image

文生图与图生图共用本模块：传 images（参考图 URL）即为图生图。
"""
import asyncio
import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()


def _strip_secret(value):
    """去掉环境变量值首尾的引号（防止部署环境原样注入引号导致 401 invalid_api_key）。"""
    if not value:
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


logger = logging.getLogger(__name__)

# ---- 端点配置 ----
# multimodal-generation 生图端点；如需走百炼 MaaS 工作空间端点，用 QWEN_IMAGE_BASE_URL 覆盖
QWEN_IMAGE_API_URL = os.getenv(
    "QWEN_IMAGE_BASE_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
).rstrip("/")
# 万相（wan）官方走百炼 MaaS 工作空间端点（URL 含 WorkspaceId），与千问公共端点不同；
# 调用 wan 模型前需在 .env 配置 WAN_IMAGE_BASE_URL，如：
#   https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
WAN_IMAGE_API_URL = os.getenv("WAN_IMAGE_BASE_URL", QWEN_IMAGE_API_URL).rstrip("/")
QWEN_API_KEY = _strip_secret(os.getenv("QWEN_API_KEY"))

# ---- 轮询配置：指数退避（1s → 2s → 4s → 8s → 10s 封顶）----
POLL_INITIAL_INTERVAL_SEC = 1.0
POLL_MAX_INTERVAL_SEC = 10.0
MAX_POLL_TIME_SEC = 300.0        # 轮询总超时（生图一般 1-3 分钟）
# 单次 HTTP 请求超时：同步调用需等待生图完成（1-3 分钟），读超时给足余量；
# 异步提交/轮询都是短请求，read 超时只作为兜底上限
REQUEST_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0)

# ---- 各模型分辨率映射：前端 imageProportion（宽高比） → 接口 size（宽*高）----
# 每个模型的分辨率默认值和范围都不同，禁止写统一映射，必须按模型单独定义。

# 万相 wan2.7 系列
WAN_SIZE_MAP = {
    "1:1": "2048*2048",
    "3:4": "1728*2368",
    "4:3": "2368*1728",
    "16:9": "2688*1536",
    "9:16": "1536*2688",
}

# 千问 qwen-image-2.0 系列：输出总像素需在 512*512 ~ 2048*2048 之间，默认分辨率 2048*2048（1:1）
QWEN_IMAGE_2_SIZE_MAP = {
    "1:1": "2048*2048",    # 默认值
    "3:4": "1728*2368",
    "4:3": "2368*1728",
    "16:9": "2688*1536",
    "9:16": "1536*2688",
}

# 千问 qwen-image-3.0 系列：size 为可选，未指定时由模型根据提示词自动推荐分辨率
# （T2I 像素面积 512*512 ~ 2048*2048，宽高比限制 1:8 ~ 8:1）
QWEN_IMAGE_3_SIZE_MAP = {
    "1:1": "2048*2048",
    "3:4": "1728*2368",
    "4:3": "2368*1728",
    "16:9": "2688*1536",
    "9:16": "1536*2688",
}

_TERMINAL_STATUS = {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}


def _auth_headers() -> dict:
    if not QWEN_API_KEY:
        raise RuntimeError("缺少 QWEN_API_KEY 环境变量，无法调用千问生图")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
    }


def get_size(size: str, size_map: dict) -> str:
    """获取分辨率：前端宽高比 → 接口 size（宽*高），未命中回退 1:1"""
    return size_map.get(size, size_map["1:1"])


def _extract_sync_image_urls(output: dict) -> list[str]:
    """从任务返回的 output 提取图片 URL（兼容两种返回格式）：
    - choices 风格：output.choices[].message.content[].image（wan / qwen-2.0 / qwen-3.0）
    - results 风格：output.results[].url（旧版异步查询）
    """
    urls = []
    for choice in output.get("choices") or []:
        msg = choice.get("message") or {}
        for item in msg.get("content") or []:
            if isinstance(item, dict) and item.get("image"):
                urls.append(item["image"])
    if not urls:
        urls = [item.get("url") for item in (output.get("results") or []) if item.get("url")]
    return urls


async def _poll_task(
    client: httpx.AsyncClient, api_url: str, headers: dict, task_id: str, model_name: str
) -> list[str]:
    """指数退避轮询任务状态，SUCCEEDED 后返回图片 URL 列表。

    轮询端点：{host}/api/v1/tasks/{task_id}（与提交端点同 host）。
    """
    parsed = urlparse(api_url)
    tasks_url = f"{parsed.scheme}://{parsed.netloc}/api/v1/tasks/{task_id}"
    delay = POLL_INITIAL_INTERVAL_SEC
    elapsed = 0.0
    while elapsed < MAX_POLL_TIME_SEC:
        await asyncio.sleep(delay)
        elapsed += delay
        resp = await client.get(tasks_url, headers=headers)
        resp.raise_for_status()
        output = (resp.json().get("output")) or {}
        status = output.get("task_status", "UNKNOWN")
        if status == "SUCCEEDED":
            urls = _extract_sync_image_urls(output)
            logger.info("生图完成（轮询） model=%s 生成%d张", model_name, len(urls))
            return urls
        if status in _TERMINAL_STATUS - {"SUCCEEDED"}:
            raise RuntimeError(f"生图任务失败 task_id={task_id} status={status} output={output}")
        delay = min(delay * 2, POLL_MAX_INTERVAL_SEC)
    raise TimeoutError(f"生图任务超时 task_id={task_id}")


async def generate_qwen_images(
    model_name: str,
    prompt: str,
    *,
    params: dict[str, Any],
    images: Any = None,
) -> list[str]:
    """调用千问 qwen-image / 万相 wan 生图（文生图/图生图共用）。

    流程：组装公共/各模型参数 → 组装请求体 → 异步提交拿 task_id → 指数退避轮询。

    参数：
        model_name:  生图模型名（如 qwen-image-3.0-pro / wan2.7-image-pro）
        prompt:      绘图提示词
        params:      前端原始参数（imageCount / imageProportion 等）
        images:      参考图 URL 列表（图生图专用），已在 generate.py 统一转成 ["url1","url2"]

    返回：
        模型生成的图片 URL 列表（原始返回，未转存业务端）。
    """
    # 1) 组装参数：公共配置 + 各模型专属配置（每个模型分辨率默认值/范围都不同，各用各的映射）
    imgQty = int(params.get("imageCount") or 1)  # 生成图片数量
    size = params.get("imageProportion", "1:1")  # 分辨率/宽高比
    base_config = {
        "n": imgQty,  # 生成图片数量
        "watermark": False,  # 水印
        "negative_prompt": "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    }
    if model_name.startswith("wan"):
        # 万相2.7：专属分辨率映射（wan 不支持 prompt_extend）
        api_url = WAN_IMAGE_API_URL
        base_config["size"] = get_size(size, WAN_SIZE_MAP)  # 分辨率（宽 x 高）
    elif model_name.startswith("qwen-image-3.0"):
        # 千问3.0系列：size 可选，未指定时由模型根据提示词自动推荐分辨率
        api_url = QWEN_IMAGE_API_URL
        if params.get("imageProportion"):
            base_config["size"] = get_size(size, QWEN_IMAGE_3_SIZE_MAP)
    elif model_name.startswith("qwen-image-2.0"):
        # 千问2.0系列：官方推荐分辨率，默认 2048*2048（1:1）
        api_url = QWEN_IMAGE_API_URL
        base_config["size"] = get_size(size, QWEN_IMAGE_2_SIZE_MAP)  # 分辨率（宽 x 高）
        base_config["prompt_extend"] = True  # 官方参数：自动扩写提示词
    else:
        api_url = QWEN_IMAGE_API_URL

    # 2) 组装请求体：参考图已在 generate.py 转成 ["url1","url2"]，直接按官方 curl 格式赋值；
    #    content 中 image 在前、text 在后，多图就是多个 image 对象
    content: list[dict] = [{"image": url} for url in (images or [])]
    content.append({"text": prompt})
    payload = {
        "model": model_name,
        "input": {
            "messages": [
                {"role": "user", "content": content},
            ]
        },
        "parameters": base_config,
    }

    headers = _auth_headers()
    # 异步头只对 wan 生效：wan 走百炼 MaaS 端点支持异步（task_id + 轮询）；
    # qwen-image 系列当前 Key 不支持异步调用，必须同步等结果（官方 curl 即同步格式）
    if model_name.startswith("wan"):
        headers["X-DashScope-Async"] = "enable"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(api_url, headers=headers, json=payload)
        if resp.status_code >= 400:
            # 带上响应体，便于定位真实原因（模型 ID / 参数 / 端点问题）
            raise RuntimeError(
                f"生图请求失败 url={api_url} model={model_name} "
                f"status={resp.status_code} body={resp.text}"
            )
        body = resp.json()
        output = body.get("output") or {}
        task_id = output.get("task_id")
        if task_id:
            logger.info("生图任务已提交 model=%s task_id=%s", model_name, task_id)
            return await _poll_task(client, api_url, headers, task_id, model_name)
        # qwen 同步返回 / 个别端点未走异步：直接解析 output（results[].url 或 choices[].message.content[].image）
        urls = _extract_sync_image_urls(output)
        if not urls:
            raise RuntimeError(f"生图请求失败：{body}")
        logger.info("生图完成（同步返回） model=%s 生成%d张", model_name, len(urls))
        return urls
