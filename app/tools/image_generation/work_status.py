import logging
from typing import Literal, Optional

from app.api.java_client import API
from pydantic import BaseModel

logger = logging.getLogger(__name__)

statusType = Literal["generating", "completed", "failed", "pending"]

# 状态到API路径的映射
STATUS_PATH_MAP = {
    "generating": "/generating",# 生成中
    "completed": "/completed",# 已完成
    "failed": "/failed",# 失败
    "pending": "/pending",# 待处理
}
BASE_PATH = "/api/ai/work"
FILE_BASE_PATH = "/file"
async def change_work_status(work_id: str, status: statusType,data=None):
    """
    更新工作状态
    """
    if not work_id or not status or status not in STATUS_PATH_MAP:
        return None
    api_path = STATUS_PATH_MAP.get(status, "/generating")
    try:
        payload = {
            "id": work_id,
        }
        if status == "pending":
            # 处理请求载荷
            if isinstance(data, list):
                payload.pop({"selectList": data})


        await API.put(f"{BASE_PATH}{api_path}",json=payload)
    except Exception as e:
        logger.error(f"更新工作状态失败: work_id={work_id}, status={status}, error={e}")
        return None





# 请求体模型，按需设置哪些字段可选
class UpdateWorkImage(BaseModel):
    resultUrl: Optional[str] = None
    dataList: Optional[list[dict[str, str]]] = None
    prompt: Optional[str] = None


async def update_work_image(
    work_id: str,
    params: Optional[UpdateWorkImage] = None,
    *,
    resultUrl: Optional[str] = None,
    dataList: Optional[list[dict[str, str]]] = None,
    prompt: Optional[str] = None,
):
    """
    更新图片工作记录，自动过滤 None 字段。

    两种调用方式（二选一）：
      A) 传 Pydantic 对象（批量/结构化场景）：
           await update_work_image(work_id, UpdateWorkImage(prompt="..."))
      B) 直接传关键词参数（节点里写起来最短）：
           await update_work_image(work_id, prompt="...")
           await update_work_image(work_id, resultUrl="...", prompt="...")
    """
    if not work_id:
        return None
    api_path = "/update"
    try:
        # 合并 Pydantic 对象与关键词参数，自动过滤 None 字段
        merged = UpdateWorkImage(
            resultUrl=resultUrl if resultUrl is not None else (params.resultUrl if params else None),
            dataList=dataList if dataList is not None else (params.dataList if params else None),
            prompt=prompt if prompt is not None else (params.prompt if params else None),
        )
        payload = merged.model_dump(exclude_none=True)
        payload["id"] = work_id

        return await API.put(f"{BASE_PATH}{api_path}", json=payload)
    except Exception as e:
        logger.error(f"更新图片失败: work_id={work_id}, error={e}")
        return None

async def upload_file_by_base64(work_id: str,files: list[dict[str, str]] | dict[str, str], userId: str, ):
    """
    上传文件
    """
    if not files:
        return None

    try:
        if isinstance(files, list):
            # 批量上传
            api_path = "/uploadFileByBase64Batch"
            # 确保字段名与Java DTO一致
            file_list = []
            for file in files:
                file_list.append({
                    "base64": file["base64"] if isinstance(file["base64"], str) else file["base64"].decode(),
                    "fileName": file["fileName"]
                })
            payload = {
                "files": file_list,
                "userId": userId,
            }
        else:
            # 单文件上传
            api_path = "/uploadFileByBase64"
            payload = {
                "base64": files["base64"] if isinstance(files["base64"], str) else files["base64"].decode(),
                "fileName": files["fileName"],
                "userId": userId,
            }

        response = await API.post(f"{BASE_PATH}{api_path}", json=payload)
        if response.get("success"):
            if isinstance(files, list):
                # 批量上传：将所有图片URL存入dataList（格式与Java DTO一致：[{"url": ...}]）
                file_urls = [{"url": item["fileUrl"]} for item in response["data"]]
                await update_work_image(work_id, UpdateWorkImage(dataList=file_urls))
            else:
                # 单文件上传：只更新resultUrl
                await update_work_image(work_id, UpdateWorkImage(resultUrl=response["data"]["fileUrl"]))
        return response.get("data")
    except Exception as e:
        logger.error(f"上传文件失败: files={files}, error={e}")
        return None

def _extract_url_and_name(file, index=0):
    """从 Image 对象或 dict 中提取 url 和 fileName"""
    if hasattr(file, 'url'):
        # OpenAI Image 对象
        url = file.url
        name = getattr(file, 'fileName', None) or f"image_{index}.png"
    elif isinstance(file, dict):
        url = file.get('url', '')
        name = file.get('fileName', f"image_{index}.png")
    else:
        url = str(file)
        name = f"image_{index}.png"
    return url, name

async def upload_file_by_url(work_id: str, files: list | dict | str, userId: str):
    """
    上传文件URL，支持 OpenAI Image 对象、dict、str
    """
    if not files:
        return None

    try:
        if isinstance(files, list):
            # 批量上传
            api_path = "/uploadFileByUrlBatch"
            file_list = []
            for i, file in enumerate(files):
                url, name = _extract_url_and_name(file, i)
                file_list.append({"url": url, "fileName": name})
            payload = {"files": file_list, "userId": userId}
        elif isinstance(files, str):
            # 单个URL字符串
            api_path = "/uploadFileByUrl"
            payload = {"url": files, "userId": userId}
        else:
            # 单个 Image 对象或 dict
            api_path = "/uploadFileByUrl"
            url, name = _extract_url_and_name(files)
            payload = {"url": url, "fileName": name, "userId": userId}

        response = await API.post(f"{FILE_BASE_PATH}{api_path}", json=payload)
        if response.get("success"):
            data = response["data"]
            if isinstance(files, list):
                # 批量上传：data是列表，提取每个fileUrl（格式与Java DTO一致：[{"url": ...}]）
                file_urls = [{"url": item["fileUrl"]} for item in data]
                await update_work_image(work_id, UpdateWorkImage(dataList=file_urls))
            else:
                # 单文件上传：data是dict，提取fileUrl
                await update_work_image(work_id, UpdateWorkImage(resultUrl=data["fileUrl"]))
        return response.get("data")
    except Exception as e:
        logger.error(f"上传文件URL失败: error={e}")
        return None


if __name__ == "__main__":
    import asyncio

    test_url = 'https://ark-acg-cn-beijing.tos-cn-beijing.volces.com/doubao-seedream-5-0/021784962151433fc13f77dd38da39bcdb33ad9bd570640de0d17_0.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTYWJkZTExNjA1ZDUyNDc3YzhjNTM5OGIyNjBhNDcyOTQ%2F20260725%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Date=20260725T064936Z&X-Tos-Expires=86400&X-Tos-Signature=5572d98d5c59580045fe9684c6242d4c3fe5a00462c4147a3891fa950b9b5fd5&X-Tos-SignedHeaders=host'
    work_id = "74391036306295616"
    user_id = "66095266633409024"

    # 模拟 OpenAI Image 对象（和火山引擎返回的格式一致）
    class MockImage:
        def __init__(self, url):
            self.url = url
            self.b64_json = None
            self.revised_prompt = None
            self.size = '2048x2048'

    async def test():
        # 测试1：单个 Image 对象
        print("--- 测试单个 Image 对象 ---")
        img = MockImage(test_url)
        result = await upload_file_by_url(work_id, img, user_id)
        print(f"单个结果: {result}")

        # 测试2：Image 对象列表（批量）
        print("\n--- 测试 Image 对象列表 ---")
        images = [MockImage(test_url)]
        result = await upload_file_by_url(work_id, images, user_id)
        print(f"批量结果: {result}")

        # 测试3：纯 URL 字符串
        print("\n--- 测试纯 URL 字符串 ---")
        result = await upload_file_by_url(work_id, test_url, user_id)
        print(f"字符串结果: {result}")

    asyncio.run(test())
