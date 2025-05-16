# audit_app/app_utils.py
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# 导入UI模型和核心模型
from AssessmentSystem.audit_app.models_ui import (
    UIAssessmentReport,
    UIAssessmentItem,
    UIEvidence,
    ItemAuditStatus,
    Judgement, # 从 models_ui 导入，它会处理自己的 fallback
    EvidenceSearchParams,
    CoreConclusion
)

# 默认报告路径
DEFAULT_INPUT_REPORT_DIR = "Data\\Report\\"
DEFAULT_INPUT_REPORT_FILENAME = "assessment_report_example.json"
REVIEWED_REPORT_SUFFIX = "_reviewed"

def load_assessment_report(file_path: str) -> Optional[UIAssessmentReport]:
    """
    从指定的JSON文件加载评估报告.
    此函数可以加载原始的AI生成报告或之前保存的已审核报告。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 报告文件未找到于 {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 解析JSON文件失败 {file_path}")
        return None

    ui_assessment_items = []
    raw_assessment_results = raw_data.get("assessment_results", [])

    for raw_item in raw_assessment_results:
        # 解析 initial_search_params
        raw_search_params = raw_item.get("evidence_search_params")
        initial_search_params = None
        if raw_search_params and isinstance(raw_search_params, dict):
            initial_search_params = EvidenceSearchParams(**raw_search_params)

        # 解析 initial_conclusion (AI的结论)
        raw_conclusion = raw_item.get("conclusion")
        initial_conclusion_obj = None
        raw_initial_evidences_list = [] # 存储原始证据字典列表

        if raw_conclusion and isinstance(raw_conclusion, dict):
            # 尝试从核心模型转换 judgement
            try:
                initial_judgement = Judgement(raw_conclusion.get("judgement", Judgement.NOT_PROCESSED.value))
            except ValueError:
                initial_judgement = Judgement.ERROR # 如果值无效，则标记为错误

            initial_comment = raw_conclusion.get("comment", "")
            raw_initial_evidences_list = raw_conclusion.get("evidence", [])

            initial_conclusion_obj = CoreConclusion(
                judgement=initial_judgement,
                comment=initial_comment,
                evidence=raw_initial_evidences_list # CoreConclusion.evidence 期望字典列表
            )

        # 初始化当前审核状态的字段，如果正在加载已审核的报告，这些值将被覆盖
        current_judgement_val = initial_conclusion_obj.judgement if initial_conclusion_obj else Judgement.NOT_PROCESSED
        current_comment_val = initial_conclusion_obj.comment if initial_conclusion_obj else ""

        # 从原始证据列表创建 UIEvidence 对象列表，用于初始化 referenced_evidences
        # 如果正在加载已审核的报告，并且证据项包含 is_active_for_conclusion，则使用它
        current_referenced_evidences = []
        for ev_dict in raw_initial_evidences_list:
            ui_ev = UIEvidence.from_report_evidence_dict(ev_dict)
            # 如果加载的是已审核报告，它可能包含 is_active_for_conclusion 字段
            if "is_active_for_conclusion" in ev_dict:
                ui_ev.is_active_for_conclusion = ev_dict["is_active_for_conclusion"]
            current_referenced_evidences.append(ui_ev)

        # 检查是否有已保存的审核员结论 (用于加载已审核的报告)
        # 我们约定在保存时，审核员的结论会覆盖原始的 'conclusion' 字段，
        # 或者我们可以在保存的条目中添加 'auditor_conclusion' 这样的字段。
        # 为了简化，这里假设如果 'judgement' 和 'comment' 在顶层 raw_item 中，它们是审核员的。
        # 更稳健的方法是在保存时明确区分AI结论和审核员结论。
        # 当前模型 UIAssessmentItem 将 current_judgement/comment 用于审核员的输入。

        # 尝试加载已保存的审核员判断和评论 (如果存在于 raw_item 级别，覆盖从 conclusion 初始化的值)
        # 这是为了兼容一种可能的保存格式，其中审核员的结论直接在 assessment_item 级别
        if "current_judgement" in raw_item:
             try:
                current_judgement_val = Judgement(raw_item["current_judgement"])
             except ValueError:
                current_judgement_val = Judgement.ERROR
        if "current_comment" in raw_item:
            current_comment_val = raw_item["current_comment"]

        # 如果在已审核报告中，referenced_evidences 是直接保存的
        if "referenced_evidences" in raw_item and isinstance(raw_item["referenced_evidences"], list):
            current_referenced_evidences = []
            for rev_ev_dict in raw_item["referenced_evidences"]:
                 # 假设 rev_ev_dict 已经是 UIEvidence 的字典表示
                current_referenced_evidences.append(UIEvidence(**rev_ev_dict))


        # 加载审核状态
        audit_status_str = raw_item.get("audit_status", ItemAuditStatus.NOT_REVIEWED.value)
        try:
            audit_status = ItemAuditStatus(audit_status_str)
        except ValueError:
            audit_status = ItemAuditStatus.NOT_REVIEWED # 值无效则回退

        ui_item = UIAssessmentItem(
            spec_id=raw_item.get("spec_id", "未知ID"),
            spec_content=raw_item.get("spec_content", ""),
            initial_search_params=initial_search_params,
            initial_conclusion=initial_conclusion_obj,
            ai_assessment_status=raw_item.get("status"), # AI评估的状态
            ai_error_message=raw_item.get("error_message"),
            current_judgement=current_judgement_val,
            current_comment=current_comment_val,
            referenced_evidences=current_referenced_evidences,
            audit_status=audit_status,
            raw_initial_evidences=raw_initial_evidences_list # 存储原始证据字典
        )
        ui_assessment_items.append(ui_item)

    # 加载报告元数据 (如原始的 statics)
    report_metadata = {k: v for k, v in raw_data.items() if k != "assessment_results" and k != "saved_current_item_index"}

    # 加载上次保存的索引
    saved_current_item_index = raw_data.get("saved_current_item_index", 0)
    # 确保索引在有效范围内
    if not (0 <= saved_current_item_index < len(ui_assessment_items)) and ui_assessment_items:
        saved_current_item_index = 0


    ui_report = UIAssessmentReport(
        report_metadata=report_metadata,
        assessment_items=ui_assessment_items,
        current_item_index=saved_current_item_index,
        original_file_path=file_path
    )
    ui_report.update_stats() # 初始化统计数据
    return ui_report


def save_assessment_report(report_data: UIAssessmentReport, output_dir: str) -> Optional[str]:
    """
    将审核后的评估报告 (UIAssessmentReport) 保存到JSON文件。
    文件名将包含当前日期。
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"创建目录 {output_dir} 失败: {e}")
            return None

    # 从原始文件名生成已审核报告的文件名
    original_filename = os.path.basename(report_data.original_file_path or DEFAULT_INPUT_REPORT_FILENAME)
    name_part, ext_part = os.path.splitext(original_filename)

    # 移除可能的旧 _reviewed_date 后缀，以防重复添加
    if REVIEWED_REPORT_SUFFIX in name_part:
        name_part = name_part.split(REVIEWED_REPORT_SUFFIX)[0]

    timestamp = datetime.now().strftime("%y-%m-%d") # YY-MM-DD 格式
    reviewed_filename = f"{name_part}{REVIEWED_REPORT_SUFFIX}_{timestamp}{ext_part}"
    output_path = os.path.join(output_dir, reviewed_filename)
    report_data.reviewed_file_path = output_path

    # 构建要保存的字典
    output_dict = report_data.report_metadata.copy() # 从原始元数据开始

    # 添加/更新审核后的评估结果
    output_dict["assessment_results"] = []
    for item in report_data.assessment_items:
        # 将 UIEvidence 转换回字典列表，包含 is_active_for_conclusion
        # 这是为了保存引用证据的激活/抑制状态
        saved_referenced_evidences = []
        for ui_ev in item.referenced_evidences:
            saved_referenced_evidences.append({
                "evidence_id": ui_ev.evidence_id, # 或者 "source": ui_ev.evidence_id
                "source": ui_ev.evidence_id, # 保持与 assessment_report_example.json 一致的字段名
                "content": ui_ev.content,
                "title": ui_ev.title,
                "evidence_type": ui_ev.evidence_type,
                "timestamp_str": ui_ev.timestamp_str,
                "is_active_for_conclusion": ui_ev.is_active_for_conclusion,
                "search_tags": ui_ev.search_tags
            })

        # 评估结论使用审核员的当前结论
        item_conclusion_dict = {
            "judgement": item.current_judgement.value,
            "comment": item.current_comment,
            "evidence": saved_referenced_evidences # 保存处理过的证据列表
        }

        item_dict = {
            "spec_id": item.spec_id,
            "spec_content": item.spec_content,
            "evidence_search_params": item.initial_search_params.model_dump() if item.initial_search_params else None,
            "conclusion": item_conclusion_dict, # 使用审核员的结论和处理过的证据
            "status": item.ai_assessment_status, # 保留原始AI评估状态
            "error_message": item.ai_error_message, # 保留原始AI错误信息
            "audit_status": item.audit_status.value, # 保存审核状态
            # 如果需要，也可以保存 current_judgement 和 current_comment 在 item_dict 的顶层，
            # 但将它们放在 "conclusion" 中更符合原始结构。
            # "current_judgement": item.current_judgement.value, (可选的冗余保存)
            # "current_comment": item.current_comment, (可选的冗余保存)
            "referenced_evidences": [ev.model_dump() for ev in item.referenced_evidences] # (可选) 更详细地保存UIEvidence对象
        }
        output_dict["assessment_results"].append(item_dict)

    # 更新统计数据
    report_data.update_stats() # 确保是最新的
    output_dict["statics_judgement"] = report_data.stats_judgement
    output_dict["statics_audit_status"] = report_data.stats_audit_status

    # 保存当前审核索引以便恢复
    output_dict["saved_current_item_index"] = report_data.current_item_index

    # 保留原始报告中 'statics' 字段（如果存在且与新的统计字段不冲突）
    if "statics" not in output_dict and "statics" in report_data.report_metadata:
        output_dict["statics"] = report_data.report_metadata["statics"]


    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)
        print(f"报告已保存到: {output_path}")
        return output_path
    except IOError as e:
        print(f"保存报告到 {output_path} 失败: {e}")
        return None

def get_default_report_path() -> str:
    """获取默认的输入报告文件路径"""
    # 脚本的当前工作目录可能不是项目根目录，因此需要更可靠的路径解析
    # 假设此脚本位于 audit_app 目录，Data 在其父目录的同级
    return os.path.join(DEFAULT_INPUT_REPORT_DIR.strip(os.path.sep), DEFAULT_INPUT_REPORT_FILENAME)


if __name__ == '__main__':
    # 测试加载和保存功能
    print("测试 app_utils.py...")

    # 尝试加载示例报告
    example_report_path = get_default_report_path()
    print(f"尝试从以下路径加载示例报告: {example_report_path}")

    loaded_report = load_assessment_report(example_report_path)

    if loaded_report:
        print(f"成功加载报告: {loaded_report.original_file_path}")
        print(f"共 {len(loaded_report.assessment_items)} 个评估项。")
        print(f"当前索引: {loaded_report.current_item_index}")

        current_item = loaded_report.get_current_assessment_item()
        if current_item:
            print(f"当前评估项 ID: {current_item.spec_id}")
            current_item.current_comment = "这是一个由测试代码修改的审核评论。"
            current_item.current_judgement = Judgement.NON_COMPLIANT
            current_item.audit_status = ItemAuditStatus.REVIEWED
            if current_item.referenced_evidences:
                current_item.referenced_evidences[0].is_active_for_conclusion = False # 禁用第一个证据


            # 模拟修改并保存
            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "Report", "test_outputs")
            saved_path = save_assessment_report(loaded_report, save_dir)
            if saved_path:
                print(f"测试报告已保存到: {saved_path}")
                # 尝试重新加载已保存的报告
                reloaded_report = load_assessment_report(saved_path)
                if reloaded_report:
                    print("重新加载已保存的报告成功。")
                    reloaded_item = reloaded_report.get_current_assessment_item()
                    if reloaded_item:
                        print(f"重新加载的评估项评论: {reloaded_item.current_comment}")
                        print(f"重新加载的评估项判断: {reloaded_item.current_judgement.value}")
                        print(f"重新加载的评估项审核状态: {reloaded_item.audit_status.value}")
                        if reloaded_item.referenced_evidences:
                             print(f"重新加载的第一个证据的激活状态: {reloaded_item.referenced_evidences[0].is_active_for_conclusion}")

        else:
            print("报告中没有评估项。")
    else:
        print("加载报告失败。")

    print("\n测试默认报告路径:")
    print(get_default_report_path())

