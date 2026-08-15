"""
结构化表登记服务（表目录 + 行数据）
==================================
解决"多张表（n 个产品表 / m 个人员部门表）时检索怎么知道查哪张表"：
- kb_structured_tables：表目录 registry（每张表一条：列名/列类型/样例行/摘要/归属），
  体积小，检索时全量加载作为 LLM 选表上下文
- kb_structured_rows：行级结构化数据（每个产品/人员一条），
  供 query_table / count_rows / list_rows 工具做确定性筛选/统计/排序

写入时机（入库阶段一次性完成，检索时零成本读取）：
- 入库编排（knowledge_base_doc_service.ingest_document）解析 Excel/docx 表格后调用 save_tables_for_doc
- force_reingest 重建 / 文档删除 时调用 delete_by_doc_ids 联动清理

归属过滤：所有查询都带 kb_id/owner_id/folder_id，与向量检索同一套权限隔离。
"""
import asyncio
import datetime
import hashlib
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient

from app.config.settings import settings
from app.utils.logger_handle import logger


# ---------------------------------------------------------------------------
# 列名 / 数值 处理（MongoDB 字段名不允许 '.' 与 '$'，统一替换，正常中文列名零影响）
# ---------------------------------------------------------------------------

def sanitize_key(name: Any) -> str:
    return str(name).replace(".", "_").replace("$", "_")


def _to_num(v: Any):
    """宽松转数值：数字原样返回；字符串去千分位/百分号后尝试转 float。"""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("%", "")
        try:
            f = float(s)
            return int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            return None
    return None


def _infer_column_types(rows: List[Dict[str, Any]], columns: List[str]) -> Dict[str, str]:
    """按列统计：非空值 ≥3 且 ≥80% 可转数值 → number，否则 string（返回键为 sanitize 后列名）。"""
    types: Dict[str, str] = {}
    for col in columns:
        num_cnt = 0
        non_empty = 0
        for r in rows:
            v = r.get(col)
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            non_empty += 1
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                num_cnt += 1
            elif isinstance(v, str):
                try:
                    float(v.strip().replace(",", "").replace("%", ""))
                    num_cnt += 1
                except (TypeError, ValueError):
                    pass
        key = sanitize_key(col)
        types[key] = "number" if non_empty >= 3 and num_cnt / non_empty >= 0.8 else "string"
    return types


def _build_summary(doc_name: str, sheet_name: str, columns: List[str], sample: Dict[str, Any]) -> str:
    """规则版表摘要（零成本）：文档名 + Sheet 名 + 列名 + 样例，足够 LLM 选表。"""
    cols = "、".join(columns[:8]) + ("…" if len(columns) > 8 else "")
    base = f"{doc_name or ''}({sheet_name or ''})：列名[{cols}]"
    if sample:
        sample_str = "；".join(f"{k}:{v}" for k, v in list(sample.items())[:5])
        return f"{base}，样例[{sample_str}]"
    return base


# ---------------------------------------------------------------------------
# 结构化表登记服务（单例）
# ---------------------------------------------------------------------------

class StructuredTableRegistry:
    """
    kb_structured_tables: {
        table_id: "doc_id:Sheet1"（唯一，主键索引）,
        doc_id, doc_name, sheet_name, source_type: "excel"|"docx",
        kb_id, owner_id, folder_id,
        columns: [sanitize 后列名], raw_columns: [原始列名],
        column_types: {"价格": "number", ...},
        sample: {"价格": 1299, ...},
        summary: "...", row_count, chunk_count, created_at, updated_at,
    }
    kb_structured_rows: {
        table_id, doc_id, kb_id, owner_id, row_index,
        data: {sanitize 列名: 原始值},          // 完整行（供 LLM 组织话术）
        num__价格: 1299,                          // 数值列副本（可比较/排序）
        cat__适用场景: "钓鱼",                    // 短文本列副本（等值/包含过滤）
        created_at,
    }
    """

    _instance = None
    _client: MongoClient = None
    _tables_col = None
    _rows_col = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_init(self):
        if self._tables_col is not None:
            return
        self._client = MongoClient(settings.mongodb_url, **settings.mongodb_conn_kwargs)
        db = self._client[settings.mongodb_db_name]
        self._tables_col = db["kb_structured_tables"]
        self._rows_col = db["kb_structured_rows"]
        self._tables_col.create_index([("table_id", ASCENDING)], unique=True, background=True)
        self._tables_col.create_index([("doc_id", ASCENDING)], background=True)
        self._tables_col.create_index([("kb_id", ASCENDING), ("owner_id", ASCENDING)], background=True)
        self._rows_col.create_index([("table_id", ASCENDING)], background=True)
        self._rows_col.create_index([("doc_id", ASCENDING)], background=True)
        self._rows_col.create_index([("kb_id", ASCENDING), ("owner_id", ASCENDING)], background=True)

    # -------- sync 内部实现（asyncio.to_thread 用） --------

    def _delete_by_doc_ids_sync(self, doc_ids: List[str]) -> Dict[str, int]:
        t = self._tables_col.delete_many({"doc_id": {"$in": doc_ids}}).deleted_count
        r = self._rows_col.delete_many({"doc_id": {"$in": doc_ids}}).deleted_count
        return {"tables": t, "rows": r}

    def _save_tables_sync(
        self,
        *,
        doc_id: str,
        doc_name: str,
        kb_id: str,
        owner_id: str,
        folder_id: Any,
        source_type: str,
        tables: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        now = datetime.datetime.now().isoformat()
        # 幂等重建：先清旧表（含行数据）
        self._delete_by_doc_ids_sync([doc_id])

        t_inserted = 0
        r_inserted = 0
        for tbl in tables:
            sheet_name = str(tbl.get("sheet_name") or "Sheet")
            raw_columns = [str(c) for c in (tbl.get("columns") or [])]
            rows = tbl.get("rows") or []
            if not raw_columns or not rows:
                continue
            columns = [sanitize_key(c) for c in raw_columns]
            table_id = f"{doc_id}:{sanitize_key(sheet_name)}"

            column_types = _infer_column_types(rows, raw_columns)

            # 样例行：第一行非空行（数据值用原始列名键）
            sample: Dict[str, Any] = {}
            for r in rows:
                sample = {c: r.get(c) for c in raw_columns if r.get(c) is not None}
                if sample:
                    break
            summary = _build_summary(doc_name, sheet_name, raw_columns, sample)

            self._tables_col.replace_one(
                {"table_id": table_id},
                {
                    "table_id": table_id,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "sheet_name": sheet_name,
                    "source_type": source_type,
                    "kb_id": kb_id,
                    "owner_id": owner_id,
                    "folder_id": folder_id,
                    "columns": columns,
                    "raw_columns": raw_columns,
                    "column_types": column_types,
                    "sample": sample,
                    "summary": summary,
                    "row_count": len(rows),
                    "chunk_count": int(tbl.get("chunk_count") or len(rows)),
                    "created_at": now,
                    "updated_at": now,
                },
                upsert=True,
            )
            t_inserted += 1

            # 行数据
            bulk = []
            for row_idx, r in enumerate(rows):
                doc_row: Dict[str, Any] = {
                    "table_id": table_id,
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "owner_id": owner_id,
                    "row_index": row_idx,
                    "data": {},
                    "created_at": now,
                }
                for raw_col in raw_columns:
                    col = sanitize_key(raw_col)
                    v = r.get(raw_col)
                    if v is None:
                        continue
                    doc_row["data"][col] = v
                    num = _to_num(v)
                    if num is not None:
                        doc_row[f"num__{col}"] = num
                    elif isinstance(v, str) and 0 < len(v) <= 64:
                        doc_row[f"cat__{col}"] = v
                bulk.append(doc_row)
            if bulk:
                self._rows_col.insert_many(bulk, ordered=False)
                r_inserted += len(bulk)

        return {"tables": t_inserted, "rows": r_inserted}

    def _get_table_sync(self, table_id: str) -> Optional[Dict[str, Any]]:
        return self._tables_col.find_one({"table_id": table_id}, {"_id": 0})

    def _list_tables_sync(
        self,
        *,
        kb_id: str = None,
        owner_id: str = None,
        folder_id: Any = None,
        keyword: str = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if kb_id:
            query["kb_id"] = kb_id
        if owner_id:
            query["owner_id"] = owner_id
        if folder_id is not None:
            query["folder_id"] = folder_id
        if keyword:
            query["$or"] = [
                {"summary": {"$regex": keyword, "$options": "i"}},
                {"doc_name": {"$regex": keyword, "$options": "i"}},
            ]
        return list(
            self._tables_col.find(query, {"_id": 0})
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )

    def _resolve_column(self, columns: List[str], key: str) -> Optional[str]:
        """把条件键解析为表内列名（sanitize 后）。
        支持：精确 → 去 max_/min_ 前缀后精确 → 包含匹配。找不到返回 None。"""
        k = sanitize_key(key)
        if k in columns:
            return k
        stripped = k
        for p in ("max_", "min_"):
            if stripped.startswith(p):
                stripped = stripped[len(p):]
                break
        if stripped in columns:
            return stripped
        for c in columns:
            if c in stripped or stripped in c:
                return c
        return None

    def _build_filter_sync(self, table_id: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """按 conditions 构造 rows collection 查询条件（表存在性+列名校验+条件翻译复用）。"""
        table = self._get_table_sync(table_id)
        if not table:
            raise ValueError(f"表不存在: {table_id}")
        columns = table.get("columns") or []
        column_types = table.get("column_types") or {}
        query: Dict[str, Any] = {"table_id": table_id}
        for raw_key, val in (conditions or {}).items():
            if val is None or val == "":
                continue
            col = self._resolve_column(columns, raw_key)
            if not col:
                raise ValueError(f"表 {table_id} 中不存在列: {raw_key}（可用列: {', '.join(table.get('raw_columns') or columns)}）")
            is_num = column_types.get(col) == "number"
            key = sanitize_key(raw_key)
            if is_num and (key.startswith("max_") or key.startswith("min_")):
                op = "$lte" if key.startswith("max_") else "$gte"
                num_val = _to_num(val)
                if num_val is not None:
                    query[f"num__{col}"] = {op: num_val}
                continue
            if is_num:
                num_val = _to_num(val)
                if num_val is not None:
                    query[f"num__{col}"] = num_val
                else:
                    query[f"cat__{col}"] = str(val)
                continue
            s = str(val).strip()
            if not s:
                continue
            if s.startswith("~"):
                query[f"cat__{col}"] = {"$regex": s[1:], "$options": "i"}
            else:
                query[f"cat__{col}"] = s
        return query

    def _query_rows_sync(
        self,
        table_id: str,
        conditions: Dict[str, Any],
        *,
        top_n: int = 20,
        order_by: str = None,
        order_dir: str = "asc",
    ) -> Dict[str, Any]:
        table = self._get_table_sync(table_id)
        if not table:
            raise ValueError(f"表不存在: {table_id}")
        columns = table.get("columns") or []
        column_types = table.get("column_types") or {}
        query = self._build_filter_sync(table_id, conditions)

        cursor = self._rows_col.find(query, {"_id": 0, "created_at": 0})
        if order_by:
            col = self._resolve_column(columns, order_by)
            if col and column_types.get(col) == "number":
                cursor = cursor.sort(f"num__{col}", ASCENDING if order_dir != "desc" else DESCENDING)
        rows = list(cursor.limit(max(1, min(int(top_n or 20), 500))))
        return {
            "table_id": table_id,
            "total_matched": self._rows_col.count_documents(query),
            "returned": len(rows),
            "rows": rows,
        }

    def _count_rows_sync(self, table_id: str, conditions: Dict[str, Any]) -> int:
        query = self._build_filter_sync(table_id, conditions)
        return self._rows_col.count_documents(query)

    def _distinct_values_sync(
        self,
        table_id: str,
        column: str,
        conditions: Dict[str, Any],
        *,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """某列去重枚举（"产品有哪些种类/品牌/产地"精确回答），按出现次数降序。"""
        table = self._get_table_sync(table_id)
        if not table:
            raise ValueError(f"表不存在: {table_id}")
        columns = table.get("columns") or []
        col = self._resolve_column(columns, column)
        if not col:
            raise ValueError(f"表 {table_id} 中不存在列: {column}")
        query = self._build_filter_sync(table_id, conditions)
        pipeline = [
            {"$match": query},
            {"$group": {"_id": f"$data.{col}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
            {"$limit": max(1, min(int(limit or 200), 1000))},
        ]
        raw = list(self._rows_col.aggregate(pipeline))
        values, counts = [], []
        for item in raw:
            v = item.get("_id")
            if v is None:
                continue
            values.append(v)
            counts.append({"value": v, "count": item.get("count", 0)})
        return {
            "table_id": table_id,
            "column": col,
            "values": values,
            "counts": counts,
            "total_distinct": len(values),
        }

    def _aggregate_stats_sync(
        self,
        table_id: str,
        group_by: str,
        conditions: Dict[str, Any],
        *,
        agg_column: str = None,
        agg_op: str = "count",
        limit: int = 200,
    ) -> Dict[str, Any]:
        """分组统计（"每个种类各有多少个产品"），按统计值降序。"""
        table = self._get_table_sync(table_id)
        if not table:
            raise ValueError(f"表不存在: {table_id}")
        columns = table.get("columns") or []
        column_types = table.get("column_types") or {}
        g_col = self._resolve_column(columns, group_by)
        if not g_col:
            raise ValueError(f"表 {table_id} 中不存在分组列: {group_by}")
        query = self._build_filter_sync(table_id, conditions)

        agg_op = (agg_op or "count").lower()
        if agg_column:
            a_col = self._resolve_column(columns, agg_column)
            if not a_col:
                raise ValueError(f"表 {table_id} 中不存在聚合列: {agg_column}")
            is_num = column_types.get(a_col) == "number"
            a_field = f"$num__{a_col}" if is_num else f"$data.{a_col}"
        else:
            a_field = None

        if agg_op == "count":
            group_expr: Dict[str, Any] = {"value": {"$sum": 1}}
        elif agg_op in ("sum", "avg") and a_field:
            group_expr = {"value": {f"${agg_op}": a_field}}
        elif agg_op == "max" and a_field:
            group_expr = {"value": {"$max": a_field}}
        elif agg_op == "min" and a_field:
            group_expr = {"value": {"$min": a_field}}
        else:
            raise ValueError(f"不支持的聚合: op={agg_op} column={agg_column}")

        pipeline = [
            {"$match": query},
            {"$group": {"_id": f"$data.{g_col}", **group_expr}},
            {"$sort": {"value": -1, "_id": 1}},
            {"$limit": max(1, min(int(limit or 200), 1000))},
        ]
        raw = list(self._rows_col.aggregate(pipeline))
        groups = []
        for item in raw:
            k = item.get("_id")
            if k is None:
                continue
            groups.append({"group": k, "value": item.get("value")})
        return {
            "table_id": table_id,
            "group_by": g_col,
            "agg_column": agg_column,
            "agg_op": agg_op,
            "groups": groups,
            "total_groups": len(groups),
        }

    # -------- async 对外接口 --------

    async def save_tables_for_doc(
        self,
        *,
        doc_id: str,
        doc_name: str,
        kb_id: str = "default",
        owner_id: str = "",
        folder_id: Any = None,
        source_type: str = "excel",
        tables: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """入库阶段登记一张文档的结构化表（表目录 + 行数据），幂等重建。"""
        self._ensure_init()
        return await asyncio.to_thread(
            self._save_tables_sync,
            doc_id=doc_id,
            doc_name=doc_name,
            kb_id=kb_id,
            owner_id=owner_id,
            folder_id=folder_id,
            source_type=source_type,
            tables=tables,
        )

    async def list_tables(
        self,
        *,
        kb_id: str = None,
        owner_id: str = None,
        folder_id: Any = None,
        keyword: str = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """表目录列表（LLM 选表上下文），按归属过滤。"""
        self._ensure_init()
        return await asyncio.to_thread(
            self._list_tables_sync,
            kb_id=kb_id,
            owner_id=owner_id,
            folder_id=folder_id,
            keyword=keyword,
            limit=limit,
        )

    async def get_table(self, table_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_init()
        return await asyncio.to_thread(self._get_table_sync, table_id)

    async def query_rows(
        self,
        table_id: str,
        conditions: Dict[str, Any] = None,
        *,
        top_n: int = 20,
        order_by: str = None,
        order_dir: str = "asc",
    ) -> Dict[str, Any]:
        """按条件筛选表数据（数值范围 max_/min_ 前缀、等值、~包含），确定性执行。"""
        self._ensure_init()
        return await asyncio.to_thread(
            self._query_rows_sync,
            table_id,
            conditions or {},
            top_n=top_n,
            order_by=order_by,
            order_dir=order_dir,
        )

    async def count_rows(self, table_id: str, conditions: Dict[str, Any] = None) -> int:
        """统计满足条件的行数（"现在有几个产品"精确回答）。"""
        self._ensure_init()
        return await asyncio.to_thread(self._count_rows_sync, table_id, conditions or {})

    async def distinct_values(
        self,
        table_id: str,
        column: str,
        conditions: Dict[str, Any] = None,
        *,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """某列去重枚举（"产品有哪些种类/品牌/产地"精确回答），按出现次数降序。"""
        self._ensure_init()
        return await asyncio.to_thread(
            self._distinct_values_sync,
            table_id,
            column,
            conditions or {},
            limit=limit,
        )

    async def aggregate_stats(
        self,
        table_id: str,
        group_by: str,
        conditions: Dict[str, Any] = None,
        *,
        agg_column: str = None,
        agg_op: str = "count",
        limit: int = 200,
    ) -> Dict[str, Any]:
        """分组统计（"每个种类各有多少个产品"），按统计值降序。agg_op: count/sum/avg/max/min"""
        self._ensure_init()
        return await asyncio.to_thread(
            self._aggregate_stats_sync,
            table_id,
            group_by,
            conditions or {},
            agg_column=agg_column,
            agg_op=agg_op,
            limit=limit,
        )

    async def delete_by_doc_ids(self, doc_ids: List[str]) -> Dict[str, int]:
        """文档删除/重新入库时联动清理表数据。"""
        if not doc_ids:
            return {"tables": 0, "rows": 0}
        self._ensure_init()
        return await asyncio.to_thread(self._delete_by_doc_ids_sync, doc_ids)


TABLE_REGISTRY = StructuredTableRegistry()
