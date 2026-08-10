import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


# 分点符号 + 数字/字母编号列表：触发 paragraph 切分，不改变 heading 层级
# 覆盖：实心圆点/空心圆点/方形点/黑点/破折号/短横线 + 数字 1. 1) 1）1、 + 字母 A. A) A. + 圆圈 ①②③ + 括号 ⑴⑵⑶
_BULLET_LINE_REGEX = re.compile(
    r'^\s*'
    r'(?:'
    # --- 圆点/方形/符号型项目符号
    r'[●●•・◦○■□▪▫⦿⦾⚫⚪※✓✔☑☒✗✘]'
    # --- 短横线/破折号（注意 -/–/—/―，后面不能接数字（不然是范围号 1-2）
    r'|[-–—―](?=\s+[^\d])'
    # --- 数字编号：1./1./1./1、/1)/1) + 两位数
    r'|\d{1,2}[.．、)]'
    r'|）\d{1,2}[.．、)]'  # 全角右括号开头
    # --- 圆圈数字：①-⑳ + ⑴-⒇ + ⒈-⒛ + ㊀-㊉（中文数字圈）
    r'|[①-⑳⑴-⒇⒈-⒛㊀-㊉]'
    # --- 罗马数字（小写 i. ii. iii. iv. v.）
    r'|[ivx]{1,4}[.．、)]'
    # --- 字母编号：A. B. C. 或 a. b. c. 或 A) a)
    r'|[A-Za-z][.．、)]'
    r')'
    r'\s+'
    r'(.+)$'
)

# 匹配纯数字 heading "1. / 2. / 3. / 10. ..."（当前识别为 h3），用于连续 2 条以上时自动降级为 bullet
_NUMERIC_H3_REGEX = re.compile(r'^\d{1,2}\.\s+(.+)$')


# ---------------------------------------------------------------------------
# 文档分类：纯规则 0 成本，用于将来按 doc_category 过滤，显著提检索精度
#   product   : 产品参数表（Excel/DOCX 表格含 型号/价格/品牌/规格/重量 列名，命中 >=2 个直接定 product
#   policy    : 政策/法规（含 条款/第X条/适用范围/罚则/依据 关键词或列名）
#   faq       : 常见问答（含 问：/答：/Q:/A:/FAQ 关键词）
#   report    : 报告/白皮书（doc_name 含 报告/白皮书/调研）
#   contract  : 合同（doc_name 含 合同/协议，或正文含 甲方/乙方/违约责任）
#   tutorial  : 教程/操作手册（doc_name 含 手册/教程/指南，或正文含 步骤/第X步）
#   text      : 兜底通用文本
# ---------------------------------------------------------------------------
_POLICY_KW = ("条款", "条", "适用范围", "罚则", "依据", "法规", "政策", "规定", "办法", "条例")
_FAQ_KW = ("问：", "答：", "Q:", "A:", "FAQ", "常见问题", "问题解答")
_REPORT_KW = ("报告", "白皮书", "调研", "研究", "分析")
_CONTRACT_KW = ("合同", "协议", "甲方", "乙方", "违约", "保证金", "盖章", "签字")
_TUTORIAL_KW = ("手册", "教程", "指南", "步骤")
_PRODUCT_COL_KW = ("型号", "价格", "品牌", "规格", "重量", "尺寸", "防水等级", "续航", "产地", "SKU", "类目")


def _guess_doc_category(
    doc_name: str,
    *,
    table_columns: List[str] = None,
    title: str = "",
    text_hint: str = "",
) -> str:
    name = (doc_name or "")
    title = (title or "")
    hint = (text_hint or "")

    # 1. doc_name 命中（最强特征）
    if any(k in name for k in ("合同", "协议")) or "contract" in name.lower():
        return "contract"
    if any(k in name for k in _REPORT_KW) or "whitepaper" in name.lower():
        return "report"
    if any(k in name for k in _FAQ_KW) or "faq" in name.lower():
        return "faq"
    if any(k in name for k in _TUTORIAL_KW) or "manual" in name.lower() or "guide" in name.lower():
        return "tutorial"
    if any(k in name for k in ("政策", "法规", "条例", "办法", "规定")):
        return "policy"

    # 2. 表格列名命中（产品参数表特征非常强）
    cols = [str(c) for c in (table_columns or [])]
    col_joined = " ".join(cols)
    if sum(1 for k in _PRODUCT_COL_KW if k in col_joined) >= 2:
        return "product"

    # 3. title / 开头文本命中
    head = (title + " " + hint)[:500]
    if sum(1 for k in _CONTRACT_KW if k in head) >= 2:
        return "contract"
    if sum(1 for k in _POLICY_KW if k in head) >= 2:
        return "policy"
    if any(k in head for k in _FAQ_KW):
        return "faq"
    if any(k in head for k in _TUTORIAL_KW):
        return "tutorial"

    # 4. Excel/DOCX 有表格且无法归类，默认 product 概率大（电商场景 Excel 大多是产品表）
    if cols:
        return "product"
    return "text"


class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # 数据清洗：移除换行符、多个空行、制表符等特殊字符
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append('')
                continue
            if re.match(r'^第[一二三四五六七八九十\d]+页$', stripped):
                continue
            if re.match(r'^\d+\s*[-–—]\s*\d+$', stripped):
                continue
            cleaned_lines.append(stripped)

        return '\n'.join(cleaned_lines).strip()

    # 数据脱敏：将手机号、身份证号、邮箱、银行卡号等敏感信息替换为占位符
    def desensitize(self, text: str) -> Tuple[str, Dict[str, int]]:
        stats = {"phone": 0, "id_card": 0, "email": 0, "bank_card": 0}

        phone_pattern = r'1[3-9]\d{9}'
        text, count = re.subn(phone_pattern, '***手机号***', text)
        stats["phone"] = count

        id_pattern = r'[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]'
        text, count = re.subn(id_pattern, '***身份证号***', text)
        stats["id_card"] = count

        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        text, count = re.subn(email_pattern, '***邮箱***', text)
        stats["email"] = count

        bank_pattern = r'\b\d{16,19}\b'
        text, count = re.subn(bank_pattern, '***银行卡号***', text)
        stats["bank_card"] = count

        return text, stats

    # 数据结构提取：从文本中提取标题、章节、节、小节等结构信息
    def extract_structure(self, text: str) -> Dict[str, Any]:
        structure = {"title": "", "headings": [], "sections": []}

        lines = text.split('\n')

        if lines and lines[0].strip():
            structure["title"] = lines[0].strip()

        heading_patterns = [
            (r'^第[一二三四五六七八九十百千\d]+章\s+(.+)$', 'h1'),
            (r'^第[一二三四五六七八九十百千\d]+节\s+(.+)$', 'h2'),
            (r'^[一二三四五六七八九十]+、\s*(.+)$', 'h2'),
            (r'^\d+\.\d+\s+(.+)$', 'h2'),
            (r'^\d+\.\s+(.+)$', 'h3'),
            (r'^（[一二三四五六七八九十]+）\s*(.+)$', 'h3'),
            (r'^\(\d+\)\s*(.+)$', 'h4'),
        ]

        current_h1 = ""
        current_h2 = ""
        current_h3 = ""

        for line in lines:
            stripped = line.strip()
            matched = False

            for pattern, level in heading_patterns:
                m = re.match(pattern, stripped)
                if m:
                    heading_text = m.group(1).strip()
                    heading_info = {
                        "text": heading_text,
                        "level": level,
                        "full_text": stripped,
                    }
                    structure["headings"].append(heading_info)

                    if level == "h1":
                        current_h1 = heading_text
                        current_h2 = ""
                        current_h3 = ""
                    elif level == "h2":
                        current_h2 = heading_text
                        current_h3 = ""
                    elif level == "h3":
                        current_h3 = heading_text

                    section_path = " > ".join(
                        filter(None, [current_h1, current_h2, current_h3])
                    )
                    structure["sections"].append({
                        "path": section_path,
                        "heading": heading_text,
                        "level": level,
                    })
                    matched = True
                    break

            if not matched:
                pass

        return structure

    # 语义分割：根据标题、章节、节、小节 + bullet 分点符号将文本分割为多个段落
    def semantic_split(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        structure = self.extract_structure(text)
        chunks = []

        lines = text.split('\n')
        current_section = metadata.get("section", "")

        heading_patterns = [
            (re.compile(r'^第[一二三四五六七八九十百千\d]+章\s+(.+)$'), 'h1'),
            (re.compile(r'^第[一二三四五六七八九十百千\d]+节\s+(.+)$'), 'h2'),
            (re.compile(r'^[一二三四五六七八九十]+、\s*(.+)$'), 'h2'),
            (re.compile(r'^\d+\.\d+\s+(.+)$'), 'h2'),
            (re.compile(r'^\d+\.\s+(.+)$'), 'h3'),
        ]

        paragraph_chunks = []
        current_paragraph = ""
        current_path_parts = []

        # -------- 连续数字 h3 降级计数器：连上 2 次 "1. xxx""2. xxx" → 全部降级为 bullet（因为真正的 h3 不可能连续编号一大段）
        prev_was_numeric_h3 = False
        pending_h3_heading_line = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # 空行打断连续 h3 计数（但不清空 pending）
                current_paragraph += "\n"
                prev_was_numeric_h3 = False
                continue

            is_heading = False
            heading_level = None
            heading_text = ""

            for pattern, level in heading_patterns:
                m = pattern.match(stripped)
                if m:
                    is_heading = True
                    heading_level = level
                    heading_text = m.group(1).strip()
                    break

            # ---- 降级处理：上一条是数字 h3，当前也是数字 h3 → 说明其实是分点列表，两条都降级为 bullet
            current_is_numeric_h3 = bool(
                is_heading and heading_level == "h3" and _NUMERIC_H3_REGEX.match(stripped)
            )
            if prev_was_numeric_h3 and current_is_numeric_h3 and pending_h3_heading_line:

                is_heading = False
                current_paragraph = (pending_h3_heading_line + "\n" + current_paragraph).strip() + "\n"
                pending_h3_heading_line = ""
                prev_was_numeric_h3 = False
            elif not current_is_numeric_h3:
                # 打断连续计数：清空 pending
                pending_h3_heading_line = ""
                prev_was_numeric_h3 = False

            if is_heading:
                # 上一条 paragraph 打包
                if current_paragraph.strip():
                    section_path = " > ".join(filter(None, current_path_parts))
                    paragraph_chunks.append({
                        "text": current_paragraph.strip(),
                        "section": section_path,
                    })
                    current_paragraph = ""

                if heading_level == "h1":
                    current_path_parts = [heading_text]
                elif heading_level == "h2":
                    if len(current_path_parts) >= 1:
                        current_path_parts = current_path_parts[:1]
                    current_path_parts.append(heading_text)
                elif heading_level == "h3":
                    if len(current_path_parts) >= 2:
                        current_path_parts = current_path_parts[:2]
                    current_path_parts.append(heading_text)

                current_paragraph = stripped + "\n"
                # 标记：当前 h3 如果是纯数字编号，pending 住以便下一条也是数字 h3 时降级
                if current_is_numeric_h3:
                    pending_h3_heading_line = stripped
                    prev_was_numeric_h3 = True
                else:
                    pending_h3_heading_line = ""
                    prev_was_numeric_h3 = False
                continue

            # ---- 非 heading：先判断是不是 bullet 分点
            m_bullet = _BULLET_LINE_REGEX.match(stripped)
            if m_bullet:
                # 遇到新 bullet → 前一段完整打包（不切 heading 层级，current_path_parts 不变）
                if current_paragraph.strip():
                    section_path = " > ".join(filter(None, current_path_parts))
                    paragraph_chunks.append({
                        "text": current_paragraph.strip(),
                        "section": section_path,
                    })
                    current_paragraph = ""
                current_paragraph = stripped + "\n"
                prev_was_numeric_h3 = False
                pending_h3_heading_line = ""
                continue

            # ---- 普通内容行：直接累加
            current_paragraph += stripped + "\n"

        if current_paragraph.strip():
            section_path = " > ".join(filter(None, current_path_parts))
            paragraph_chunks.append({
                "text": current_paragraph.strip(),
                "section": section_path or current_section,
            })

        final_chunks = []
        current_chunk_text = ""
        current_chunk_section = ""
        chunk_index = 0

        for para in paragraph_chunks:
            para_text = para["text"]
            para_section = para["section"]

            # 当前 para 塞得进当前 chunk → 合并
            if len(current_chunk_text) + len(para_text) + 2 <= self.chunk_size:
                if current_chunk_text:
                    current_chunk_text += "\n\n"
                current_chunk_text += para_text
                current_chunk_section = para_section or current_chunk_section
                continue

            # 塞不下 → 如果当前 chunk 非空 → 先打包当前 chunk
            if current_chunk_text:
                chunk_text = current_chunk_text.strip()
                content_hash = self.compute_hash(chunk_text)
                # chunk_id 绑定内容 hash：内容不同则 id 必不同（避免不同表格/段落撞同 id），
                # 内容相同则 id 相同（同 doc 重试时 upsert 幂等覆盖）
                chunk_id = hashlib.md5(
                    f"{metadata.get('doc_id', '')}_{content_hash}".encode()
                ).hexdigest()[:12]
                final_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "section": current_chunk_section,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "content_hash": content_hash,
                    }
                })
                chunk_index += 1
                current_chunk_text = ""
                current_chunk_section = para_section

            # 当前 paragraph 单条太长 → 走句子级分拆
            if len(para_text) > self.chunk_size * 1.5:
                sub_chunks = self._split_long_text(
                    para_text, para_section, metadata, chunk_index
                )
                final_chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                current_chunk_text = ""
                current_chunk_section = para_section
            else:
                # 新 chunk 起点：整条 para 直接放
                current_chunk_text = para_text
                current_chunk_section = para_section

        if current_chunk_text.strip():
            chunk_text = current_chunk_text.strip()
            content_hash = self.compute_hash(chunk_text)
            chunk_id = hashlib.md5(
                f"{metadata.get('doc_id', '')}_{content_hash}".encode()
            ).hexdigest()[:12]
            final_chunks.append({
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "section": current_chunk_section,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "content_hash": content_hash,
                }
            })

        return final_chunks

    # 长文本分割：将过长的段落分割为多个段落，每个段落不超过指定大小
    def _split_long_text(
        self, text: str, section: str, metadata: Dict[str, Any], start_index: int
    ) -> List[Dict[str, Any]]:
        chunks = []
        sentences = re.split(r'([。！？.!?\n])', text)
        current_text = ""
        chunk_index = start_index

        i = 0
        while i < len(sentences):
            part = sentences[i]
            if i + 1 < len(sentences) and sentences[i + 1] in "。！？.!?\n":
                part += sentences[i + 1]
                i += 2
            else:
                i += 1

            if len(current_text) + len(part) <= self.chunk_size:
                current_text += part
            else:
                if current_text:
                    chunk_text = current_text.strip()
                    content_hash = self.compute_hash(chunk_text)
                    chunk_id = hashlib.md5(
                        f"{metadata.get('doc_id', '')}_{content_hash}".encode()
                    ).hexdigest()[:12]
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            **metadata,
                            "section": section,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_index,
                            "content_hash": content_hash,
                        }
                    })
                    chunk_index += 1

                    overlap_text = current_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_text = overlap_text + part
                else:
                    current_text = part

        if current_text.strip():
            chunk_text = current_text.strip()
            content_hash = self.compute_hash(chunk_text)
            chunk_id = hashlib.md5(
                f"{metadata.get('doc_id', '')}_{content_hash}".encode()
            ).hexdigest()[:12]
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "section": section,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "content_hash": content_hash,
                }
            })

        return chunks

    # 计算哈希值：对文本内容进行哈希处理，用于唯一标识文本段落
    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 文档最终处理：数据清洗-》脱敏-》结构提取-》语义分割-》哈希计算
    def process_document(
        self,
        content: str,
        doc_name: str,
        doc_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        doc_id = doc_id or hashlib.md5(doc_name.encode()).hexdigest()[:12]

        cleaned_text = self.clean_text(content)

        desensitized_text, desensitize_stats = self.desensitize(cleaned_text)

        structure = self.extract_structure(desensitized_text)

        # 文档类型：纯规则猜，用于将来按 doc_category 过滤，显著提检索精度
        guessed_doc_type = _guess_doc_category(
            doc_name,
            title=structure.get("title", "") or "",
            text_hint=desensitized_text[:800],
        )

        base_metadata: Dict[str, Any] = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_category": guessed_doc_type,
            "doc_title": structure.get("title") or "",
            # extra_metadata 里通常有 folder_id / kb_id，这些是检索过滤刚需
            **(extra_metadata or {}),
        }
        if base_metadata.get("doc_title") is None:
            base_metadata["doc_title"] = ""

        chunks = self.semantic_split(desensitized_text, base_metadata)

        unique_chunks = []
        seen_hashes = set()
        duplicate_count = 0

        for chunk in chunks:
            content_hash = chunk["metadata"]["content_hash"]
            if content_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(content_hash)
            # 每个 chunk 存字符数，用于将来 rerank 权重参考（0 成本）
            if "char_count" not in chunk["metadata"]:
                chunk["metadata"]["char_count"] = len(chunk.get("content", "") or "")
            unique_chunks.append(chunk)

        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "original_length": len(content),
            "cleaned_length": len(cleaned_text),
            "total_chunks": len(chunks),
            "unique_chunks": len(unique_chunks),
            "duplicate_chunks": duplicate_count,
            "desensitize_stats": desensitize_stats,
            "structure": structure,
            "chunks": unique_chunks,
            "metadata": {
                **base_metadata,
                "total_chunks": len(chunks),
                "unique_chunks": len(unique_chunks),
            },
        }

    # ============================================================
    # Excel / 结构化表格处理：标准表（首行表头），行级块化，不走语义分割
    # ============================================================

    @staticmethod
    def _is_number(v: Any):
        if v is None or v == "":
            return False
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
        if isinstance(v, str):
            s = v.strip().replace(",", "").replace("%", "")
            try:
                float(s)
                return True
            except (TypeError, ValueError):
                return False
        return False

    @staticmethod
    def _to_number(v: Any):
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            s = v.strip().replace(",", "").replace("%", "")
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
                return f
            except (TypeError, ValueError):
                return None
        return None

    def _clean_cell_value(self, raw: Any) -> str:
        """单元格值清洗：转字符串 + clean_text 清洗的子集（只清多余空白）
        """
        if raw is None:
            return ""
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            # 数字直接字符串化，但为了避免浮点显示错误，直接 str 一下让它保持默认
            return str(raw)
        s = str(raw)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # 多个空行压成一个（规格参数等长文本会折行）
        s = re.sub(r'\n{3,}', '\n\n', s)
        s = re.sub(r'[ \t]+', ' ', s)
        lines = [ln.strip() for ln in s.split("\n")]
        # 空行保留一个
        out_lines = []
        prev_empty = False
        for ln in lines:
            if ln == "":
                if prev_empty:
                    continue
                prev_empty = True
                out_lines.append("")
            else:
                prev_empty = False
                out_lines.append(ln)
        return "\n".join(out_lines).strip()

    def process_structured_rows(
        self,
        rows: List[Dict[str, Any]],
        columns: List[str],
        sheet_name: str,
        base_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        标准表 → 行级块化（每一行 = 1 个 chunk）。
        每块格式：
            【产品信息 - {title列或SKU列或第一列值】
              列名1: 值1
              列名2: 值2
        不走 clean_text / desensitize / semantic_split（因为行是天然原子块）。
        单元格值单独跑 _clean_cell_value + 敏感信息脱敏（手机号/身份证/邮箱/银行卡）。
        metadata 额外加：
            - sheet_name / row_index（0-based，不含表头）
            - 每一列在 columns 作为全量列名存一份（便于 BM25 搜列名
            - 数值列：{col}_num 存数值类型
            - 分类列：{col}_cat 存原值字符串
        """
        import json
        from copy import deepcopy

        chunks: List[Dict[str, Any]] = []

        if not rows:
            return chunks

        # 为避免 base_metadata 里有 doc_id/doc_name
        doc_id = base_metadata.get("doc_id", "")
        chunk_index_global = 0
        desensitize_total = {"phone": 0, "id_card": 0, "email": 0, "bank_card": 0}

        # 标题候选列（优先产品名/SKU，用来做 chunk 首行【产品信息 - xxx】
        title_candidates = ["产品名称", "商品名称", "名称", "title", "product_name", "商品名"]
        sku_candidates = ["SKU", "sku", "货号", "型号"]
        id_candidates = ["产品ID", "商品ID", "id", "ID"]

        def _find_col(prefer_list):
            c1 = [c for c in prefer_list if c in columns]
            return c1[0] if c1 else None

        title_col = _find_col(title_candidates) or _find_col(sku_candidates) or _find_col(id_candidates) or (columns[0] if columns else None)

        for row_idx, row in enumerate(rows):
            # 1. 先对每个单元格做清洗 + 脱敏
            cleaned_row = {}
            for col in columns:
                raw = row.get(col)
                cv = self._clean_cell_value(raw)
                if cv:
                    # 对字符串值做脱敏（数字列不要做，不然会炸手机号银行卡误匹配）
                    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                        cv, d_stats = self.desensitize(cv)
                        for k, n in d_stats.items():
                            desensitize_total[k] += n
                cleaned_row[col] = cv

            # 2. 生成 chunk 展示文本
            title_value = cleaned_row.get(title_col, "") if title_col else ""
            if title_value:
                header = f"【{sheet_name} - {title_value}】"
            else:
                header = f"【{sheet_name} - 第{row_idx + 2}行】"
            body_lines = [header]
            for col in columns:
                v = cleaned_row.get(col, "")
                if v is None or v == "":
                    # 空值也要带列名吗？不要，太乱；留着以后开关
                    continue
                body_lines.append(f"  {col}: {v}")
            chunk_text = "\n".join(body_lines).strip()

            # 3. metadata 填充：基础 + 数值列/cat 列字段
            chunk_meta: Dict[str, Any] = {
                **deepcopy(base_metadata),
                "sheet_name": sheet_name,
                "row_index": row_idx,
                "chunk_index": chunk_index_global,
                "char_count": len(chunk_text),
            }
            # 数值列
            for col in columns:
                raw = row.get(col)
                cv = cleaned_row.get(col, "")
                if self._is_number(raw) and not isinstance(raw, bool):
                    num = self._to_number(raw)
                    if num is not None:
                        chunk_meta[f"num__{col}"] = num
                # 字符串分类短文本都存 cat
                if isinstance(cv, str) and 0 < len(cv) <= 64:
                    chunk_meta[f"cat__{col}"] = cv

            # chunk_id 绑定内容 hash：不同表格可能共享同一 sheet_name（docx preceding_title 启发式），
            # 只用 sheet_name+row_idx 会撞 id；绑定内容后内容不同则 id 必不同，同内容重试则幂等覆盖
            content_hash = self.compute_hash(chunk_text)
            chunk_id = hashlib.md5(
                f"{doc_id}_{sheet_name}_{row_idx}_{content_hash}".encode()
            ).hexdigest()[:12]
            chunk_meta["chunk_id"] = chunk_id
            chunk_meta["content_hash"] = content_hash

            chunks.append({
                "content": chunk_text,
                "metadata": chunk_meta,
            })
            chunk_index_global += 1

        # 去重（按 content_hash
        seen_hashes = set()
        unique_chunks = []
        dup_cnt = 0
        for ch in chunks:
            h = ch["metadata"]["content_hash"]
            if h in seen_hashes:
                dup_cnt += 1
                continue
            seen_hashes.add(h)
            unique_chunks.append(ch)

        ch_meta = {
            "row_chunk_total": len(chunks),
            "row_chunk_unique": len(unique_chunks),
            "duplicate_chunks": dup_cnt,
            "desensitize_stats": desensitize_total,
        }
        # 简单用最后一个 chunk 带出统计
        return unique_chunks, ch_meta

    def process_excel_file(
        self,
        file_path: str,
        doc_name: str,
        doc_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        标准 Excel 文件处理入口（.xlsx / .xls。
        返回结构对齐 process_document，便于上游统一处理。
        """
        import os
        from app.utils.file_handler import excel_loader

        doc_id = doc_id or hashlib.md5(doc_name.encode()).hexdigest()[:12]

        # 1. 读结构化表（先读表，拿到 sheet/columns 信息给 doc_category 猜更准）
        structured_sheets = excel_loader(file_path, output_structured=True)
        all_columns: List[str] = []
        for sht in (structured_sheets or []):
            for col in (sht.get("columns") or []):
                all_columns.append(str(col))

        # 先猜文档分类，再塞 base_metadata，所有 chunks 继承
        guessed_doc_type = _guess_doc_category(
            doc_name, table_columns=list(set(all_columns)), title=os.path.splitext(doc_name)[0],
        )

        base_metadata = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_category": guessed_doc_type,
            "doc_title": os.path.splitext(doc_name)[0],
            "source_type": "excel",
            "has_table": True,
            **(extra_metadata or {}),
        }
        if not structured_sheets:
            return {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "original_length": 0,
                "cleaned_length": 0,
                "total_chunks": 0,
                "unique_chunks": 0,
                "duplicate_chunks": 0,
                "desensitize_stats": {"phone":0, "id_card":0, "email":0, "bank_card":0},
                "sheets": [],
                "chunks": [],
                "metadata": base_metadata,
                "sheet_summaries": [],
            }

        # 2. 每个 sheet 行级块化（表头前的前言文本先收集，循环后统一并入文档流）
        all_chunks = []
        sheet_summaries = []
        structured_tables = []  # 供入库阶段结构化登记（表目录 + 行数据）
        prefix_parts = []
        total_original_chars = 0
        total_cleaned_chars = 0
        overall_desensitize = {"phone":0, "id_card":0, "email":0, "bank_card":0}
        for sht in structured_sheets:
            sht_name = sht["sheet_name"]
            cols = sht["columns"]
            rows = sht["rows"]
            prefix_text = sht.get("prefix_text") or ""
            if prefix_text:
                prefix_parts.append(prefix_text)
            # 原始字符数粗略估计：sum(len(str(cell)))
            orig_cnt = 0
            for r in rows:
                for c in cols:
                    v = r.get(c)
                    if v is not None:
                        orig_cnt += len(str(v))
            total_original_chars += orig_cnt

            unique_chunks, ch_meta = self.process_structured_rows(rows, cols, sht_name, base_metadata)
            sheet_summaries.append({
                "sheet_name": sht_name,
                "columns": cols,
                "row_count": len(rows),
                "chunk_count": len(unique_chunks),
            })
            structured_tables.append({
                "sheet_name": sht_name,
                "columns": cols,
                "rows": rows,
                "chunk_count": len(unique_chunks),
            })
            for k, n in ch_meta["desensitize_stats"].items():
                overall_desensitize[k] += n
            total_cleaned_chars += sum(len(c["content"]) for c in unique_chunks)
            all_chunks.extend(unique_chunks)

        # 2.1 表头前的前言文本（如"本表收录在售产品 200 个"）→ 语义分块，并入文档流
        if prefix_parts:
            prefix_result = self.process_document(
                "\n\n".join(prefix_parts),
                doc_name=doc_name,
                doc_id=doc_id,
                extra_metadata=extra_metadata,
            )
            all_chunks.extend(prefix_result.get("chunks", []))
            total_original_chars += prefix_result.get("original_length", 0)
            total_cleaned_chars += prefix_result.get("cleaned_length", 0)
            for k, n in (prefix_result.get("desensitize_stats") or {}).items():
                overall_desensitize[k] = overall_desensitize.get(k, 0) + int(n or 0)

        # 3. 批次去重（跨 sheet）
        seen_final = set()
        final_chunks = []
        cross_dup = 0
        for ch in all_chunks:
            h = ch["metadata"]["content_hash"]
            if h in seen_final:
                cross_dup += 1
                continue
            seen_final.add(h)
            # 重新赋全局 chunk_index
            ch["metadata"]["chunk_index"] = len(final_chunks)
            # 确保每个 chunk 有 char_count（用于 rerank 权重参考）
            if "char_count" not in ch["metadata"]:
                ch["metadata"]["char_count"] = len(ch.get("content", "") or "")
            final_chunks.append(ch)

        # 文档级统计字段，不要往每个 chunk 里塞（纯展示）
        doc_level_meta = {
            **base_metadata,
            "sheet_count": len(sheet_summaries),
            "table_count": len(sheet_summaries),
            "total_chunks": len(all_chunks),
            "unique_chunks": len(final_chunks),
        }

        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "original_length": total_original_chars,
            "cleaned_length": total_cleaned_chars,
            "total_chunks": len(all_chunks),
            "unique_chunks": len(final_chunks),
            "duplicate_chunks": ch_meta["row_chunk_total"] - ch_meta["row_chunk_unique"] + cross_dup,
            "desensitize_stats": overall_desensitize,
            "sheets": sheet_summaries,
            "chunks": final_chunks,
            "metadata": doc_level_meta,
            "sheet_summaries": sheet_summaries,
            "structured_tables": structured_tables,  # 供入库阶段结构化登记
        }

    def process_docx_file(
        self,
        file_path: str,
        doc_name: str,
        doc_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        DOCX 处理入口：段落流 + 表格流双流水线。
        - 段落流 → process_document（clean/脱敏/结构提取/新 bullet semantic_split）
        - 表格流 → 每个表格复用 process_structured_rows（行级块化，每个表格 preceding_title 当 sheet_name）
        两路 chunks 合并后，统一 content_hash 去重返回。
        返回结构对齐 process_document/process_excel_file。
        """
        import os
        from copy import deepcopy
        from app.utils.file_handler import docx_loader

        doc_id = doc_id or hashlib.md5(doc_name.encode()).hexdigest()[:12]

        # 先读 DOCX 结构，拿到表格/段落信息给 doc_category 猜更准
        docx_data = docx_loader(file_path, output_structured=True)
        if not isinstance(docx_data, dict):
            docx_data = {"paragraph_text": "", "tables": []}
        paragraph_text = docx_data.get("paragraph_text", "") or ""
        tables = docx_data.get("tables", []) or []
        all_columns: List[str] = []
        for tbl in tables:
            for col in (tbl.get("columns") or []):
                all_columns.append(str(col))

        guessed_doc_type = _guess_doc_category(
            doc_name,
            table_columns=list(set(all_columns)),
            title=os.path.splitext(doc_name)[0],
            text_hint=paragraph_text[:800],
        )

        base_metadata: Dict[str, Any] = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_category": guessed_doc_type,
            "doc_title": os.path.splitext(doc_name)[0],
            "source_type": "docx",
            "has_table": bool(tables),
            "table_count": len(tables),
            **(extra_metadata or {}),
        }

        merged_chunks = []
        total_cleaned_chars = 0
        overall_desensitize = {"phone": 0, "id_card": 0, "email": 0, "bank_card": 0}
        table_summaries = []
        paragraph_result_stats = {
            "total_chunks": 0,
            "unique_chunks": 0,
            "original_length": len(paragraph_text),
            "cleaned_length": 0,
            "duplicate_chunks": 0,
        }

        # 2. 段落流：process_document 完整跑一遍（clean/脱敏/新 bullet semantic_split 全生效）
        if paragraph_text.strip():
            para_result = self.process_document(
                content=paragraph_text,
                doc_name=doc_name,
                doc_id=doc_id,
                extra_metadata=deepcopy(extra_metadata or {}),
            )
            para_chunks = para_result.get("chunks", [])
            merged_chunks.extend(para_chunks)
            total_cleaned_chars += para_result.get("cleaned_length", 0)
            for k, n in (para_result.get("desensitize_stats") or {}).items():
                overall_desensitize[k] = overall_desensitize.get(k, 0) + int(n or 0)
            paragraph_result_stats.update({
                "total_chunks": para_result.get("total_chunks", len(para_chunks)),
                "unique_chunks": para_result.get("unique_chunks", len(para_chunks)),
                "cleaned_length": para_result.get("cleaned_length", 0),
                "duplicate_chunks": para_result.get("duplicate_chunks", 0),
            })

        # 3. 表格流：每个表格 → process_structured_rows（preceding_title 为 sheet_name）
        structured_tables = []  # 供入库阶段结构化登记（表目录 + 行数据）
        for tbl in tables:
            tbl_idx = tbl.get("table_index", 0)
            preceding_title = tbl.get("preceding_title") or f"Table_{tbl_idx + 1}"
            cols = tbl.get("columns") or []
            rows = tbl.get("rows") or []
            unique_chunks, tbl_meta = self.process_structured_rows(
                rows, cols, preceding_title, base_metadata,
            )
            table_summaries.append({
                "sheet_name": preceding_title,
                "columns": cols,
                "row_count": len(rows),
                "chunk_count": len(unique_chunks),
                "preceding_title": preceding_title,
                "table_index": tbl_idx,
            })
            structured_tables.append({
                "sheet_name": preceding_title,
                "columns": cols,
                "rows": rows,
                "chunk_count": len(unique_chunks),
            })
            for k, n in (tbl_meta.get("desensitize_stats") or {}).items():
                overall_desensitize[k] = overall_desensitize.get(k, 0) + int(n or 0)
            total_cleaned_chars += sum(len(c["content"]) for c in unique_chunks)
            merged_chunks.extend(unique_chunks)

        # 4. 全局去重（段落与表格可能有重复内容，如标题既在段落里出现，表格里又出现同名章节）
        seen = set()
        final_chunks = []
        cross_dup = 0
        for ch in merged_chunks:
            h = ch["metadata"]["content_hash"]
            if h in seen:
                cross_dup += 1
                continue
            seen.add(h)
            ch["metadata"]["chunk_index"] = len(final_chunks)
            # 确保每个 chunk 有 char_count（用于 rerank 权重参考，0 成本）
            if "char_count" not in ch["metadata"]:
                ch["metadata"]["char_count"] = len(ch.get("content", "") or "")
            final_chunks.append(ch)

        # 给表格 chunk 附加 source_type=docx_table 标识，便于以后过滤
        for ch in final_chunks:
            m = ch["metadata"]
            if "sheet_name" in m or "row_index" in m or "table_index" in m:
                m["source"] = m.get("source") or "docx_table"
            else:
                m["source"] = m.get("source") or "docx_paragraph"

        total_dup = paragraph_result_stats.get("duplicate_chunks", 0) + cross_dup
        orig_length = paragraph_result_stats.get("original_length", 0) + sum(
            len(str(v)) for tbl in tables for r in tbl.get("rows", []) for v in r.values()
        )

        # 文档级统计字段，不要往每个 chunk 里塞（纯展示）
        doc_level_meta = {
            **base_metadata,
            "total_chunks": len(merged_chunks),
            "unique_chunks": len(final_chunks),
            "duplicate_chunks": total_dup,
        }

        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "original_length": orig_length,
            "cleaned_length": total_cleaned_chars,
            "total_chunks": len(merged_chunks),
            "unique_chunks": len(final_chunks),
            "duplicate_chunks": total_dup,
            "desensitize_stats": overall_desensitize,
            "tables": table_summaries,
            "paragraph_stats": paragraph_result_stats,
            "chunks": final_chunks,
            "metadata": doc_level_meta,
            "sheet_summaries": table_summaries,  # 与 Excel 响应保持同名，便于 Java 端统一展示
            "structured_tables": structured_tables,  # 供入库阶段结构化登记
        }
