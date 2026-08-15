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

    # ---------------------- input_check_node 输入检查输出 ----------------------
    totalScope: Optional[int]  # 描述总分 0-70
    need_manual_count: Optional[int] # 薄弱维度数量（决策核心判定参数）
    judgeList: Optional[Dict[str, int]]  # 各维度分项分数明细
    judge_summary: Optional[str] #评分结果
    # ---------------------- decision_node 方案决策输出 ----------------------
    isPass: Optional[bool]  # 是否直接放行生图 True/False
    decide_result: Optional[str] # 决策判定一句话原因

    # ---------------------- supplementary_node 补充选择题输出 ----------------------
    selectList: Optional[List[Dict[str, Any]]]  # 前端展示的补充选择题列表
    selectResult: Optional[List[Dict[str, str]]] #选择结果
    supplementary_loop_count: Optional[int]  # interrupt_node → supplementary 循环次数（超过3次自动放行）
    # ---------------------- generate_node 生图节点输出 ----------------------
    image_list: Optional[List[str]]  # 生成图片地址列表（统一为数组，支持单张/多张）
    metadata: Optional[Dict[str, Any]]  # 生图元数据

    # ---------------------- 节点重试机制 ----------------------
    node_error: Optional[str]     # 最近一次节点执行错误信息（有值说明需要重试）
    retry_target: Optional[str]   # 需要重试的节点名
    retry_count: Optional[int]    # 手动重试轮数（防死循环）
