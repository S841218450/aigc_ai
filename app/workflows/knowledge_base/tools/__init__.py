# -*- coding: utf-8 -*-
"""知识库工作流工具包：agent 通过 tool calling 查询知识库的入口。

按用途区分文件：
- doc_tools.py：文档检索（混合检索 / 文档清单）+ 检索范围辅助
- table_tools.py：结构化数据表查询（list_tables / count_rows / query_rows / distinct_values / aggregate_stats）
"""
