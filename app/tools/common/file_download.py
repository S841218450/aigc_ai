"""
文件下载 → 临时文件 → 提取纯文本 工具
======================================
知识库文档入库（file_url 模式）与问答附件（document 类型）共用的下载解析能力。
"""
import asyncio
import hashlib
import os
import tempfile
from typing import Tuple
from urllib.parse import urlparse

import httpx

from app.config.settings import settings
from app.utils.file_handler import pdf_loader, txt_loader, excel_loader, docx_loader

# 允许的文件后缀 → loader 映射，用户上传格式扩展时在这里加
FILE_LOADER_MAP = {
    ".pdf": pdf_loader,
    ".txt": txt_loader,
    ".md": txt_loader,  # Markdown 直接按纯文本处理，分割器能识别标题层级
    ".xlsx": excel_loader,
    ".xls": excel_loader,
    ".docx": docx_loader,
}

# Excel 专属后缀集合（旁路：行级块化，不走 content 拼接 + semantic_split）
EXCEL_SUFFIXES = {".xlsx", ".xls"}

# DOCX 专属后缀集合（旁路：段落流+表格流双流水线）
DOCX_SUFFIXES = {".docx"}

# 下载超时和临时目录
DOWNLOAD_TIMEOUT = 60.0  # 大文件最多等 60 秒
MAX_DOWNLOAD_MB = 50     # 单文件大小限制 50MB，超出直接拒


def cos_download_headers() -> dict:
    """COS 防盗链默认拒绝空 Referer，下载文件时带上配置的 Referer（空则不携带）"""
    referer = (settings.cos_download_referer or "").strip()
    return {"Referer": referer} if referer else {}


def detect_file_type(url: str, default_type: str = "") -> Tuple[str, str]:
    """
    根据 URL 推断文件后缀和类型。
    返回 (suffix_with_dot, file_type_without_dot)。
    例："https://xxx.com/a.PDF" → (".pdf", "pdf")
    """
    parsed = urlparse(url)
    path = parsed.path.lower()
    for ext in FILE_LOADER_MAP:
        if path.endswith(ext):
            return ext, ext.lstrip(".")
    return "", default_type.lstrip(".").lower()


async def download_and_extract_content(
    file_url: str,
    *,
    file_name_hint: str = "",
    file_type_hint: str = "",
) -> Tuple[str, str, int, str]:
    """
    从 file_url 下载文件到临时目录 → 选择对应 loader 提取纯文本。
    临时文件在函数返回前自动清理。

    Args:
        file_url: 可直接访问的文件 URL
        file_name_hint: Java 端传的 doc_name，用来兜底推断类型
        file_type_hint: Java 端传的文件类型（可选）

    Returns:
        (content: str, file_type: str, file_size_bytes: int, file_md5: str)

    Raises:
        ValueError: 不支持的类型 / 文件过大 / 下载失败 / 解析失败
    """
    if not file_url:
        raise ValueError("file_url 为空")

    suffix, file_type = detect_file_type(file_url, default_type=file_type_hint)
    if not suffix and file_name_hint:
        # URL 没后缀，用 doc_name 推断
        suffix, file_type = detect_file_type(file_name_hint, default_type=file_type_hint)

    if suffix not in FILE_LOADER_MAP:
        supported = ", ".join(sorted(FILE_LOADER_MAP.keys()))
        raise ValueError(f"暂不支持的文件类型: {suffix or file_type or '未知'}, 已支持: {supported}")

    # 下载文件到临时目录
    tmp_dir = tempfile.mkdtemp(prefix="kb_upload_")
    tmp_path = os.path.join(tmp_dir, f"download{suffix}")
    file_size = 0
    file_md5 = ""

    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", file_url, headers=cos_download_headers()) as resp:
                resp.raise_for_status()
                # Content-Length 预检，防止超大文件
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_DOWNLOAD_MB * 1024 * 1024:
                    raise ValueError(
                        f"文件过大 ({int(content_length)/1024/1024:.1f}MB)，上限 {MAX_DOWNLOAD_MB}MB"
                    )

                md5_hasher = hashlib.md5()
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
                        md5_hasher.update(chunk)
                        file_size += len(chunk)
                        if file_size > MAX_DOWNLOAD_MB * 1024 * 1024:
                            raise ValueError(f"文件流超过 {MAX_DOWNLOAD_MB}MB 上限")
                file_md5 = md5_hasher.hexdigest()

        loader = FILE_LOADER_MAP[suffix]
        try:
            if suffix == ".pdf":
                loader_kwargs = {"passwd": None}
                loaded_docs = await asyncio.to_thread(loader, tmp_path, **loader_kwargs)
            else:
                loaded_docs = await asyncio.to_thread(loader, tmp_path)
        except Exception as e:
            raise ValueError(f"文件解析失败 ({file_type}): {str(e)}") from e

        # 把 LangChain Document 列表拼成单个 content 字符串，页码信息塞进 chunk metadata 还太早，
        # 先保留 Document.metadata 里的 page/section，DocumentProcessor 会把 extra_metadata 合并。
        # 这里用 page_content 直接拼接即可。
        parts: list = []
        for d in loaded_docs:
            text = getattr(d, "page_content", "")
            if not text:
                continue
            parts.append(text.strip())
        content = "\n\n".join(parts)

        if not content.strip():
            raise ValueError("文件解析后内容为空，可能是扫描版 PDF 或加密文件")

        return content, file_type, file_size, file_md5

    finally:
        # 确保临时文件和目录被清理
        for _retry in range(3):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if os.path.isdir(tmp_dir):
                    os.rmdir(tmp_dir)
                break
            except Exception:
                import time
                time.sleep(0.05)


async def download_file_to_tmp(
    file_url: str,
    *,
    file_name_hint: str = "",
    file_type_hint: str = "",
) -> Tuple[str, str, str, str, int, str]:
    """
    仅下载文件到临时目录，不做内容解析。供 Excel/DOCX 旁路和统一入口使用。

    Returns:
        (tmp_dir, tmp_path, suffix, file_type, file_size_bytes, file_md5)
    """
    if not file_url:
        raise ValueError("file_url 为空")
    suffix, file_type = detect_file_type(file_url, default_type=file_type_hint)
    if not suffix and file_name_hint:
        suffix, file_type = detect_file_type(file_name_hint, default_type=file_type_hint)
    if suffix not in FILE_LOADER_MAP:
        supported = ", ".join(sorted(FILE_LOADER_MAP.keys()))
        raise ValueError(
            f"暂不支持的文件类型: {suffix or file_type or '未知'}, 已支持: {supported}"
        )

    tmp_dir = tempfile.mkdtemp(prefix="kb_upload_")
    tmp_path = os.path.join(tmp_dir, f"download{suffix}")
    file_size = 0
    file_md5 = ""

    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", file_url, headers=cos_download_headers()) as resp:
                resp.raise_for_status()
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_DOWNLOAD_MB * 1024 * 1024:
                    raise ValueError(
                        f"文件过大 ({int(content_length)/1024/1024:.1f}MB)，上限 {MAX_DOWNLOAD_MB}MB"
                    )
                md5_hasher = hashlib.md5()
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
                        md5_hasher.update(chunk)
                        file_size += len(chunk)
                        if file_size > MAX_DOWNLOAD_MB * 1024 * 1024:
                            raise ValueError(f"文件流超过 {MAX_DOWNLOAD_MB}MB 上限")
                file_md5 = md5_hasher.hexdigest()

        return tmp_dir, tmp_path, suffix, file_type, file_size, file_md5
    except Exception:
        # 失败立即清理临时文件
        for _r in range(3):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if os.path.isdir(tmp_dir):
                    os.rmdir(tmp_dir)
                break
            except Exception:
                import time as _t
                _t.sleep(0.05)
        raise


def cleanup_tmp(tmp_dir: str, tmp_path: str):
    for _ in range(3):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
            break
        except Exception:
            import time as _t
            _t.sleep(0.05)
