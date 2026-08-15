# -*- coding: utf-8 -*-
"""结构化数据表查询工具集：agent 通过 tool calling 查表的入口。

与文档检索分离：数据类问题（统计/枚举/明细）走确定性 MongoDB 精确筛选，零幻觉。
"""
from typing import Any, Callable, Dict, List, Optional

from app.tools.common.table_registry import TABLE_REGISTRY


def _format_conditions_desc(conditions: Dict[str, Any]) -> str:
    return "、".join(f"{k}={v}" for k, v in conditions.items())


def build_table_tools(kb_id: Optional[str], owner_id: Optional[str]) -> List[Callable]:
    """结构化表查询工具集：闭包绑定权限 scope，执行确定性 MongoDB 精确筛选，零幻觉。

    返回 5 个工具（list_tables / count_rows / query_rows / distinct_values / aggregate_stats），
    供主查询 agent 绑定；以后要扩展查数能力只需在此加工具。
    """

    async def list_tables() -> str:
        """列出当前知识库登记的结构化数据表目录（每行一张表：table_id、摘要、列名、行数）。回答数据类问题前必须先调用本工具获取目录。"""
        try:
            ts = await TABLE_REGISTRY.list_tables(kb_id=kb_id, owner_id=owner_id, limit=200)
        except Exception as e:
            return f"查询表目录失败: {e}"
        if not ts:
            return "（当前知识库没有登记任何结构化数据表）"
        lines = []
        for t in ts:
            cols = "、".join(t.get("raw_columns") or t.get("columns") or [])
            lines.append(
                f"- table_id: {t.get('table_id')} | 摘要: {t.get('summary')} "
                f"| 列: {cols} | 行数: {t.get('row_count')}"
            )
        return "\n".join(lines)

    async def count_rows(table_id: str, conditions: Dict[str, Any] = None) -> str:
        """统计某张表满足条件的行数（如"现在有几个产品"）。table_id 必须来自 list_tables；conditions 键为列名，数值范围用 max_/min_ 前缀，文本模糊用 ~关键词。"""
        try:
            n = await TABLE_REGISTRY.count_rows(table_id, conditions or {})
            table = await TABLE_REGISTRY.get_table(table_id)
            name = f"{table.get('doc_name')}({table.get('sheet_name')})" if table else table_id
            cond = _format_conditions_desc(conditions or {})
            return f"表「{name}」统计结果：共 {n} 条" + (f"（条件：{cond}）" if cond else "")
        except Exception as e:
            return f"统计失败: {e}"

    async def query_rows(table_id: str, conditions: Dict[str, Any] = None, top_n: int = 20,
                         order_by: str = None, order_dir: str = "asc") -> str:
        """查询某张表满足条件的明细行（如"列出戴尔的产品""价格低于1000的显示器"）。返回满足条件的总行数与前 N 行明细。"""
        try:
            res = await TABLE_REGISTRY.query_rows(
                table_id, conditions or {},
                top_n=max(1, min(int(top_n or 20), 500)),
                order_by=order_by, order_dir=order_dir,
            )
            table = await TABLE_REGISTRY.get_table(table_id)
            doc_name = table.get("doc_name", "") if table else ""
            sheet_name = table.get("sheet_name", "") if table else ""
            raw_columns = (table.get("raw_columns") or table.get("columns") or []) if table else []
            rows = res.get("rows", [])
            cond = _format_conditions_desc(conditions or {})
            lines = [
                f"表「{doc_name}({sheet_name})」满足条件共 {res.get('total_matched')} 条，展示前 {len(rows)} 条"
                + (f"（条件：{cond}）" if cond else "") + "："
            ]
            for i, r in enumerate(rows, 1):
                data = r.get("data", {})
                pairs = "，".join(f"{c}: {data.get(c, '')}" for c in raw_columns if c in data)
                lines.append(f"{i}. {pairs}")
            return "\n".join(lines)
        except Exception as e:
            return f"查询失败: {e}"

    async def distinct_values(table_id: str, column: str, conditions: Dict[str, Any] = None, limit: int = 200) -> str:
        """查询某张表某列的去重取值集合（如"产品有哪些种类""有哪些品牌"），按出现次数降序。"""
        try:
            dv = await TABLE_REGISTRY.distinct_values(
                table_id, column, conditions or {},
                limit=max(1, min(int(limit or 200), 1000)),
            )
            table = await TABLE_REGISTRY.get_table(table_id)
            name = f"{table.get('doc_name')}({table.get('sheet_name')})" if table else table_id
            counts = dv.get("counts", [])
            cond = _format_conditions_desc(conditions or {})
            lines = [
                f"表「{name}」中「{dv.get('column')}」去重枚举共 {dv.get('total_distinct')} 类"
                + (f"（条件：{cond}）" if cond else "") + "，按出现次数降序："
            ]
            for i, item in enumerate(counts, 1):
                lines.append(f"{i}. {item.get('value')}（{item.get('count')} 条）")
            return "\n".join(lines)
        except Exception as e:
            return f"去重枚举失败: {e}"

    async def aggregate_stats(table_id: str, group_by: str, agg_op: str = "count", agg_column: str = None,
                              conditions: Dict[str, Any] = None, limit: int = 200) -> str:
        """按某张表某列分组统计（如"每个种类各有多少个产品"），按统计值降序。agg_op: count/sum/avg/max/min。"""
        try:
            agg = await TABLE_REGISTRY.aggregate_stats(
                table_id, group_by, conditions or {},
                agg_column=agg_column, agg_op=agg_op,
                limit=max(1, min(int(limit or 200), 1000)),
            )
            table = await TABLE_REGISTRY.get_table(table_id)
            name = f"{table.get('doc_name')}({table.get('sheet_name')})" if table else table_id
            groups = agg.get("groups", [])
            agg_col_txt = f"/聚合列:{agg.get('agg_column')}" if agg.get("agg_column") else ""
            cond = _format_conditions_desc(conditions or {})
            lines = [
                f"表「{name}」按「{agg.get('group_by')}」{agg.get('agg_op')}" + agg_col_txt
                + (f"（条件：{cond}）" if cond else "")
                + f"共 {agg.get('total_groups')} 组，按统计值降序："
            ]
            for i, item in enumerate(groups, 1):
                lines.append(f"{i}. {item.get('group')}: {item.get('value')}")
            return "\n".join(lines)
        except Exception as e:
            return f"分组统计失败: {e}"

    return [list_tables, count_rows, query_rows, distinct_values, aggregate_stats]
