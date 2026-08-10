from typing import Optional, Dict, Any, List

from app.models.schemas.text_to_image import paramsType
from app.workflows.common.baseState import BaseState


class TextToImageState(BaseState):
    # ---------------------- 用户输入基础字段 ----------------------
    prompt: str    # 最终用于生图的优化提示词（补充选择题后更新）
    params: paramsType # SD生图参数
    model: str  # 模型
    # ---------------------- Agent 执行日志（每节点追加） ----------------------
    agent_log: Optional[str]  # 节点执行摘要，SSE 回传给前端

    # ---------------------- desc_code_judge_node 打分节点输出 ----------------------
    totalScope: Optional[int]  # 描述总分 0-70
    need_manual_count: Optional[int] # 薄弱维度数量（决策核心判定参数）
    judgeList: Optional[Dict[str, int]]  # 各维度分项分数明细
    judge_summary: Optional[str] #评分结果
    # ---------------------- decision_router 决策路由节点输出 ----------------------
    isPass: Optional[bool]  # 是否直接放行生图 True/False
    decide_result: Optional[str] # 决策判定一句话原因（修复拼写deside→decide）

    # ---------------------- supplementary_node 选择题补充节点输出 ----------------------
    selectList: Optional[List[Dict[str, Any]]]  # 前端展示的补充选择题列表
    selectResult: Optional[List[Dict[str, str]]] #选择结果
    supplementary_loop_count: Optional[int]  # human_interrupt → supplementary 循环次数（超过3次自动放行）
    # ---------------------- generate_image_node 生图节点输出 ----------------------
    raw_image_urls: Optional[List[str]]  # 原始图片URL，上传失败时可仅重试上传
    image_url: Optional[List[str]]  # 生成图片地址列表（统一为列表，支持多图）
    metadata: Optional[Dict[str, Any]]  # SD工具返回图片元数据
    upload_retried: Optional[int]   # 上传失败重试次数（防止无限循环）

    # ---------------------- summer_node 总结评估+重绘控制字段 ----------------------
    redraw_count: int   # 重绘次数，上限2次防止死循环，初始0
    need_redraw: Optional[bool]  # 是否需要回流重绘 True/False
    match_score: Optional[int]  # 图片与用户描述匹配度 0-10
    image_problem: Optional[str]   # 当前图片存在的缺陷
    modify_suggest: Optional[str]  # 绘图优化修改建议
    judge_note: Optional[str]   # 重绘判定备注（含次数限制逻辑）