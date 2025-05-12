# models.py
# 定义系统的数据模型
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel

class Judgement(str, Enum):
    """评估结论枚举"""
    COMPLIANT = "符合"
    NON_COMPLIANT = "不符合"
    NOT_APPLICABLE = "不涉及"
    NOT_PROCESSED = "未处理"

class AssessmentStatus(str, Enum):
    """评估状态枚举"""
    SUCCESS = "success"      # 评估成功
    TIMEOUT = "timeout"      # 评估超时
    FAIL = "fail"            # 评估失败
    RUNNING = "running"      # 评估中

class AssessmentSpecItem(BaseModel):
    """评估规范单项定义"""
    id:Optional[str]                  # 唯一标识符
    condition: str           # 条件描述
    heading: str             # 标题
    content: str             # 内容详情
    method: str = "default"  # 评估方法，默认为'default'

    class Config:
        extra = "ignore"  # 忽略JSON中的额外字段

class EvidenceMaterial(BaseModel):
    """证据材料数据结构"""
    content: str                         # 文本内容
    tags: Dict = None                         # 标签字典，默认为空
    source: Optional[str] = "unknown"         # 数据来源，默认'unknown'
    embedding: Optional[List[float]] = None   # 嵌入向量(可选)
    project: Optional[str] = None             # 所属项目(可选)
    collection_time: Optional[int] = None     # 采集时间戳(可选)
    collector: Optional[str] = None           # 采集人(可选)

class EvidenceSearchParams(BaseModel):
    """证据搜索参数数据结构"""
    query_text: str    # 查询文本，默认为空字符串
    filter_tags: List[str] = None

class EvidenceSearchResult(BaseModel):
    """证据搜索结果数据结构"""
    source: str = "unknown"  # 来源，默认'unknown'
    content: str = ""        # 内容摘要，默认为空字符串

class Conclusion(BaseModel):
    """评估结论数据结构"""
    judgement: Judgement = Judgement.NOT_PROCESSED  # 判定结果，默认'未处理'
    comment: str = ""                              # 说明注释，默认为空

class AssessmentResult(BaseModel):
    """单项评估结果数据结构"""
    spec_id: str                                  # 对应规范ID
    spec_content: str                             # 规范内容
    evidence: List[EvidenceSearchResult] = []     # 使用的证据列表
    conclusion: Conclusion = Conclusion()         # 评估结论
    status: AssessmentStatus = AssessmentStatus.RUNNING  # 状态，默认'评估中'
    error_message: Optional[str] = None           # 错误信息(可选)

class AssessmentReport(BaseModel):
    """评估报告数据结构"""
    assessment_results: List[AssessmentResult] = []  # 评估结果集合
    statics: Dict[str, int] = {                     # 统计信息
        "compliant": 0,        # 符合数量
        "non_compliant": 0,    # 不符合数量
        "not_applicable": 0,   # 不涉及数量
        "not_processed": 0      # 未处理数量
    }