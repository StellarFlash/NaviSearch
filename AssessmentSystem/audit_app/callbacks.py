# audit_app/callbacks.py
import gradio as gr # type: ignore # pylint: disable=import-error
from typing import List, Dict, Any, Optional, Tuple
import os
import traceback # 用于打印更详细的错误信息

# 相对导入
from AssessmentSystem.audit_app.models_ui import (
    UIAssessmentReport,
    UIAssessmentItem,
    UIEvidence,
    Judgement,
    ItemAuditStatus,
    EvidenceSearchParams
)
from AssessmentSystem.audit_app.app_utils import (
    load_assessment_report,
    save_assessment_report,
    get_default_report_path,
    DEFAULT_INPUT_REPORT_DIR
)
from AssessmentSystem.audit_app.ui_components import (
    update_report_detail_module_components,
    update_evidence_search_module_components,
    update_statistics_components,
    format_evidence_to_markdown # 可能在回调中直接生成HTML时用到
)

# 尝试导入核心系统模块
try:
    from AssessmentSystem.model import ( # pylint: disable=import-error
        AssessmentSpecItem as CoreAssessmentSpecItem,
        EvidenceSearchResult as CoreEvidenceSearchResult,
        Conclusion as CoreConclusion
    )
    from AssessmentSystem.llm_client import LLMAssessmentClient # pylint: disable=import-error
    from AssessmentSystem.navi_search_client import NaviSearchClient # pylint: disable=import-error
except ImportError:
    print("警告: callbacks.py - AssessmentSystem 中的核心模块导入失败。")
    CoreAssessmentSpecItem = None # type: ignore
    CoreEvidenceSearchResult = None # type: ignore
    CoreConclusion = None # type: ignore
    LLMAssessmentClient = None # type: ignore
    NaviSearchClient = None # type: ignore

# 全局客户端实例 (通常在 main.py 中初始化并传递，这里为了简化回调的编写，暂时假设可以访问)
# 在实际应用中，这些应该通过 functools.partial 注入，或者回调是某个类的方法。
# 为避免在每个回调中传递，我们可以在 main.py 中创建它们，然后让回调函数能够访问。
# 这里我们先定义函数，然后在 main.py 中绑定时处理客户端的传递。

# --- 辅助函数 ---
def _get_current_item_from_state(report_state: UIAssessmentReport) -> Optional[UIAssessmentItem]:
    """从报告状态中获取当前评估项。"""
    if report_state and 0 <= report_state.current_item_index < len(report_state.assessment_items):
        return report_state.assessment_items[report_state.current_item_index]
    return None

def _update_status_bar(message: str, is_error: bool = False) -> gr.update: # pylint: disable=no-member
    """更新状态栏消息。"""
    prefix = "错误: " if is_error else "状态: "
    print(prefix + message) # 同时打印到控制台
    return gr.update(value=prefix + message) # type: ignore

def _gather_current_item_audit_data(
    report_state: UIAssessmentReport,
    new_judgement_str: str,
    new_comment: str
) -> Tuple[Optional[UIAssessmentReport], str]:
    """
    收集当前评估项的审核数据（判断、评论）并更新到 report_state。
    返回更新后的 report_state 和一条状态消息。
    """
    current_item = _get_current_item_from_state(report_state)
    if not current_item:
        return report_state, "没有活动的评估项来保存数据。"

    try:
        current_item.current_judgement = Judgement(new_judgement_str)
    except ValueError:
        return report_state, f"无效的判断值: {new_judgement_str}"

    current_item.current_comment = new_comment

    # 如果项目之前是 "未审核" 并且做出了有效判断 (不是 "未处理")，则更新为 "已审核"
    if current_item.audit_status == ItemAuditStatus.NOT_REVIEWED and \
       current_item.current_judgement != Judgement.NOT_PROCESSED:
        current_item.audit_status = ItemAuditStatus.REVIEWED

    report_state.update_stats()
    return report_state, f"评估项 {current_item.spec_id} 的结论已在内存中更新。"

# --- 回调函数定义 ---

def handle_load_report_click(
    report_file_path_input: str, # 来自 gr.File 或 gr.Textbox 的路径
    # pylint: disable=unused-argument
    progress: Optional[gr.Progress] = None # Gradio 进度条 (如果使用)
) -> tuple:
    """
    处理“加载报告”按钮点击事件。
    加载指定路径的报告文件，或默认报告。
    """
    # progress.start() # 如果使用进度条
    report_path = report_file_path_input if report_file_path_input else get_default_report_path()

    if not report_path or not os.path.exists(report_path):
        status_update = _update_status_bar(f"报告文件路径无效或文件不存在: {report_path}", is_error=True)
        # 返回一个与所有输出组件数量相匹配的元组，用 gr.update() 更新它们
        # 这里简化，假设我们只更新状态和少数几个关键组件
        return (None, status_update) + (gr.update(),) * 15 # 估算输出组件数量

    report_data = load_assessment_report(report_path)

    if not report_data:
        status_update = _update_status_bar(f"加载报告失败: {report_path}", is_error=True)
        return (None, status_update) + (gr.update(),) * 15

    current_item = report_data.get_current_assessment_item()

    # 更新UI组件
    # 左侧面板
    report_detail_updates = update_report_detail_module_components(
        {}, # components dict - 在实际绑定时由 main.py 提供
        current_item,
        len(report_data.assessment_items),
        report_data.current_item_index
    )
    # 右侧面板 (搜索参数和结果通常基于当前项的初始值或清空)
    search_params_to_load = current_item.initial_search_params if current_item and current_item.initial_search_params else EvidenceSearchParams(query_text="", filter_tags=[])

    search_module_updates = update_evidence_search_module_components(
        current_search_query=search_params_to_load.query_text,
        current_filter_tags=search_params_to_load.filter_tags,
        recommended_tags=[], # 初始加载时不显示推荐标签
        search_results=[]    # 初始加载时不显示搜索结果
    )

    # 统计数据
    stats_updates = update_statistics_components(report_data) # type: ignore

    # 顶部导航进度
    progress_display_text = "无评估项"
    if current_item:
        progress_display_text = (
            f"正在审核: {report_data.current_item_index + 1} / {len(report_data.assessment_items)} "
            f"(ID: {current_item.spec_id}, 状态: {current_item.audit_status.value})"
        )

    status_update = _update_status_bar(f"报告 '{os.path.basename(report_path)}' 加载成功。")
    # print(status_update) # 打印当前项 (用于调试)
    # input()
    # 返回 (report_state, status_bar_update, progress_display_update, *report_detail_updates, *search_module_updates, *stats_updates)
    # pylint: disable=line-too-long
    return (
        report_data, # report_state (gr.State)
        status_update, # status_bar
        gr.update(value=progress_display_text), # progress_display
        *report_detail_updates, # 解包左侧面板的更新
        *search_module_updates, # 解包右侧面板的更新
        *stats_updates          # 解包统计数据的更新
    ) + (gr.update(),) * 16 # 补充缺少的22个输出值


def handle_prev_item_click(
    report_state: UIAssessmentReport,
    current_judgement_str: str, current_comment: str # 从UI收集当前项的结论
) -> tuple:
    """处理“上一条”按钮点击。"""
    if not report_state or not report_state.assessment_items:
        return (report_state, _update_status_bar("报告未加载或为空。", is_error=True)) + (gr.update(),) * 15 # pylint: disable=line-too-long

    # 1. 保存当前项的状态
    report_state, save_msg = _gather_current_item_audit_data(report_state, current_judgement_str, current_comment)

    # 2. 导航到上一条
    if report_state.current_item_index > 0:
        report_state.current_item_index -= 1
        status_msg = f"已导航到上一条评估项。{save_msg}"
    else:
        status_msg = f"已经是第一条评估项。{save_msg}"

    current_item = _get_current_item_from_state(report_state)

    # 更新UI
    report_detail_updates = update_report_detail_module_components(
        {}, current_item, len(report_state.assessment_items), report_state.current_item_index
    )
    search_params = current_item.initial_search_params if current_item and current_item.initial_search_params else EvidenceSearchParams(query_text="", filter_tags=[])
    search_module_updates = update_evidence_search_module_components(
        current_search_query=search_params.query_text,
        current_filter_tags=search_params.filter_tags,
        recommended_tags=[], search_results=[]
    )
    stats_updates = update_statistics_components(report_state) # type: ignore
    progress_text = (
        f"正在审核: {report_state.current_item_index + 1} / {len(report_state.assessment_items)} "
        f"(ID: {current_item.spec_id if current_item else 'N/A'}, 状态: {current_item.audit_status.value if current_item else 'N/A'})"
    )

    # pylint: disable=line-too-long
    return (
        report_state,
        _update_status_bar(status_msg),
        gr.update(value=progress_text),
        *report_detail_updates,
        *search_module_updates,
        *stats_updates
    ) + (gr.update(),) * 22  # 补充缺少的22个输出值

def handle_skip_item_click(report_state: UIAssessmentReport) -> tuple:
    """处理“跳过本条”按钮点击。"""
    if not report_state or not report_state.assessment_items:
        return (report_state, _update_status_bar("报告未加载或为空。", is_error=True)) + (gr.update(),) * 15 # pylint: disable=line-too-long

    current_item = _get_current_item_from_state(report_state)
    if current_item:
        current_item.audit_status = ItemAuditStatus.SKIPPED
        # 跳过也意味着清空当前未保存的结论
        current_item.current_judgement = Judgement.NOT_PROCESSED
        current_item.current_comment = ""
        # 清空LLM建议（如果有）
        current_item.llm_suggested_judgement = None
        current_item.llm_suggested_comment = None
        report_state.update_stats()
        skip_msg = f"评估项 {current_item.spec_id} 已标记为跳过。"
    else:
        skip_msg = "没有当前评估项可跳过。"

    # 导航到下一条 (如果有)
    if report_state.current_item_index < len(report_state.assessment_items) - 1:
        report_state.current_item_index += 1
        status_msg = f"已跳过并导航到下一条。{skip_msg}"
    elif current_item: # 是最后一条，跳过后停留在最后一条
        status_msg = f"已跳过最后一条评估项。{skip_msg}"
    else: # 报告为空
        status_msg = skip_msg


    next_item_to_display = _get_current_item_from_state(report_state)

    report_detail_updates = update_report_detail_module_components(
        {}, next_item_to_display, len(report_state.assessment_items), report_state.current_item_index
    )
    search_params = next_item_to_display.initial_search_params if next_item_to_display and next_item_to_display.initial_search_params else EvidenceSearchParams(query_text="", filter_tags=[])
    search_module_updates = update_evidence_search_module_components(
        current_search_query=search_params.query_text,
        current_filter_tags=search_params.filter_tags,
        recommended_tags=[], search_results=[]
    )
    stats_updates = update_statistics_components(report_state) # type: ignore
    progress_text = (
        f"正在审核: {report_state.current_item_index + 1} / {len(report_state.assessment_items)} "
        f"(ID: {next_item_to_display.spec_id if next_item_to_display else 'N/A'}, 状态: {next_item_to_display.audit_status.value if next_item_to_display else 'N/A'})"
    )

    # pylint: disable=line-too-long
    return (
        report_state,
        _update_status_bar(status_msg),
        gr.update(value=progress_text),
        *report_detail_updates,
        *search_module_updates,
        *stats_updates
    ) + (gr.update(),) * 22  # 补充缺少的22个输出值


def handle_next_item_click(
    report_state: UIAssessmentReport,
    current_judgement_str: str, current_comment: str
) -> tuple:
    """处理“通过并下一条”按钮点击。"""
    if not report_state or not report_state.assessment_items:
        return (report_state, _update_status_bar("报告未加载或为空。", is_error=True)) + (gr.update(),) * 15 # pylint: disable=line-too-long

    # 1. 保存当前项的状态，并标记为已审核
    report_state, save_msg = _gather_current_item_audit_data(report_state, current_judgement_str, current_comment)
    current_item_before_nav = _get_current_item_from_state(report_state) # 获取更新状态后的当前项

    if current_item_before_nav:
        if current_item_before_nav.current_judgement == Judgement.NOT_PROCESSED:
             # pylint: disable=line-too-long
            return (
                report_state,
                _update_status_bar("请先为当前评估项选择一个有效的判断结论，然后再进入下一条。", is_error=True),
                gr.update(), # progress
                *(gr.update() for _ in range(len(update_report_detail_module_components({}, None, 0, 0)))),
                *(gr.update() for _ in range(len(update_evidence_search_module_components()))),
                *(gr.update() for _ in range(len(update_statistics_components(report_state)))) # type: ignore
            )
        # 确保状态是 REVIEWED
        current_item_before_nav.audit_status = ItemAuditStatus.REVIEWED
        report_state.update_stats() # 再次更新统计数据以反映 audit_status 的变化

    # 2. 导航到下一条
    if report_state.current_item_index < len(report_state.assessment_items) - 1:
        report_state.current_item_index += 1
        status_msg = f"已保存并导航到下一条评估项。{save_msg}"
    elif current_item_before_nav: # 是最后一条
        status_msg = f"已保存最后一条评估项。审核完成！{save_msg}"
    else: # 报告为空
        status_msg = save_msg


    next_item_to_display = _get_current_item_from_state(report_state)

    report_detail_updates = update_report_detail_module_components(
        {}, next_item_to_display, len(report_state.assessment_items), report_state.current_item_index
    )
    search_params = next_item_to_display.initial_search_params if next_item_to_display and next_item_to_display.initial_search_params else EvidenceSearchParams(query_text="", filter_tags=[])
    search_module_updates = update_evidence_search_module_components(
        current_search_query=search_params.query_text,
        current_filter_tags=search_params.filter_tags,
        recommended_tags=[], search_results=[]
    )
    stats_updates = update_statistics_components(report_state) # type: ignore
    progress_text = (
        f"正在审核: {report_state.current_item_index + 1} / {len(report_state.assessment_items)} "
        f"(ID: {next_item_to_display.spec_id if next_item_to_display else 'N/A'}, 状态: {next_item_to_display.audit_status.value if next_item_to_display else 'N/A'})"
    )

    # pylint: disable=line-too-long
    return (
        report_state,
        _update_status_bar(status_msg),
        gr.update(value=progress_text),
        *report_detail_updates,
        *search_module_updates,
        *stats_updates
    ) + (gr.update(),) * 22  # 补充缺少的22个输出值

def handle_save_work_click(
    report_state: UIAssessmentReport,
    current_judgement_str: str, current_comment: str
) -> Tuple[UIAssessmentReport, gr.update]: # pylint: disable=no-member
    """处理“保存当前工作”按钮点击。"""
    if not report_state:
        return report_state, _update_status_bar("没有报告数据可保存。", is_error=True)

    # 1. 确保当前编辑的结论和评论被保存到 report_state 中
    report_state, save_msg = _gather_current_item_audit_data(report_state, current_judgement_str, current_comment)

    # 2. 执行保存操作
    output_dir = os.path.join(os.path.dirname(get_default_report_path()), "") # 保存到与输入报告同目录
    saved_file_path = save_assessment_report(report_state, output_dir)

    if saved_file_path:
        report_state.reviewed_file_path = saved_file_path # 更新状态中的保存路径
        return report_state, _update_status_bar(f"工作已保存到 {saved_file_path}。{save_msg}")
    else:
        return report_state, _update_status_bar(f"保存工作失败。{save_msg}", is_error=True)


# --- 引用证据操作回调 ---
def handle_toggle_reference_evidence(
    report_state: UIAssessmentReport,
    selected_evidence_id: str # 来自 Radio 选择器
) -> Tuple[UIAssessmentReport, gr.update, gr.update]: # pylint: disable=no-member
    """切换选中引用证据的激活/抑制状态。"""
    current_item = _get_current_item_from_state(report_state)
    status_msg = "操作失败。"
    ref_ev_display_html_update = gr.update() # type: ignore

    if current_item and selected_evidence_id:
        found = False
        for ev in current_item.referenced_evidences:
            if ev.evidence_id == selected_evidence_id:
                ev.is_active_for_conclusion = not ev.is_active_for_conclusion
                status_msg = f"证据 '{selected_evidence_id}' 的状态已切换为: {'已激活' if ev.is_active_for_conclusion else '已抑制'}。"
                found = True
                break
        if not found:
            status_msg = f"未在引用列表中找到证据ID: {selected_evidence_id}"

        # 重新生成引用证据区的HTML
        html_parts = [format_evidence_to_markdown(ev, is_referenced=True, index=i)
                      for i, ev in enumerate(current_item.referenced_evidences)]
        ref_ev_display_html_update = gr.update(value="".join(html_parts) if html_parts else "无引用证据") # type: ignore
    elif not current_item:
        status_msg = "没有当前评估项。"
    elif not selected_evidence_id:
        status_msg = "请先选择一个引用证据进行操作。"

    return report_state, _update_status_bar(status_msg), ref_ev_display_html_update

def handle_remove_reference_evidence(
    report_state: UIAssessmentReport,
    selected_evidence_id: str # 来自 Radio 选择器
) -> Tuple[UIAssessmentReport, gr.update, gr.update, gr.update]: # pylint: disable=no-member
    """从当前评估项的引用列表中移除选中的证据。"""
    current_item = _get_current_item_from_state(report_state)
    status_msg = "操作失败。"
    ref_ev_display_html_update = gr.update() # type: ignore
    ref_ev_selector_update = gr.update() # type: ignore
    ref_ev_actions_row_update = gr.update() # type: ignore

    if current_item and selected_evidence_id:
        original_len = len(current_item.referenced_evidences)
        current_item.referenced_evidences = [
            ev for ev in current_item.referenced_evidences if ev.evidence_id != selected_evidence_id
        ]
        if len(current_item.referenced_evidences) < original_len:
            status_msg = f"证据 '{selected_evidence_id}' 已从引用列表移除。"
        else:
            status_msg = f"未在引用列表中找到证据ID: {selected_evidence_id}，无法移除。"

        # 更新显示
        html_parts = [format_evidence_to_markdown(ev, is_referenced=True, index=i)
                      for i, ev in enumerate(current_item.referenced_evidences)]
        ref_ev_display_html_update = gr.update(value="".join(html_parts) if html_parts else "无引用证据") # type: ignore

        new_choices = [ev.evidence_id for ev in current_item.referenced_evidences]
        ref_ev_selector_update = gr.update(choices=new_choices, value=None, visible=bool(new_choices)) # type: ignore
        ref_ev_actions_row_update = gr.update(visible=bool(new_choices)) # type: ignore

    elif not current_item:
        status_msg = "没有当前评估项。"
    elif not selected_evidence_id:
        status_msg = "请先选择一个引用证据进行移除。"

    return report_state, _update_status_bar(status_msg), ref_ev_display_html_update, ref_ev_selector_update, ref_ev_actions_row_update


# --- 证据搜索回调 ---
_current_search_filter_tags: List[str] = [] # 模块级变量存储当前过滤标签 (简单实现)
_last_search_results: List[UIEvidence] = [] # 存储上次搜索结果，用于引用
_last_recommended_tags: List[str] = [] # 存储上次推荐标签

def handle_add_filter_tag(
    report_state: UIAssessmentReport, # pylint: disable=unused-argument
    tag_to_add: str,
    search_query_text: str # 当前搜索框内容，用于保持状态
) -> Tuple[gr.update, gr.update, gr.update]: # pylint: disable=no-member
    """添加过滤标签。"""
    global _current_search_filter_tags # pylint: disable=global-statement
    status_msg = ""
    if tag_to_add and tag_to_add.strip():
        tag = tag_to_add.strip()
        if tag not in _current_search_filter_tags:
            _current_search_filter_tags.append(tag)
            status_msg = f"标签 '{tag}' 已添加至过滤器。"
        else:
            status_msg = f"标签 '{tag}' 已存在于过滤器中。"
    else:
        status_msg = "请输入有效的标签名。"

    # 更新UI组件 (只更新搜索模块的部分)
    # pylint: disable=line-too-long
    search_module_updates = update_evidence_search_module_components(
        current_search_query=search_query_text, # 保持用户输入的查询文本
        current_filter_tags=_current_search_filter_tags,
        recommended_tags=_last_recommended_tags, # 保持上次的推荐
        search_results=_last_search_results      # 保持上次的搜索结果
    )
    # 返回 (状态栏，已选标签HTML，推荐标签HTML，搜索结果HTML，搜索结果选择器，搜索结果操作行)
    # 我们只更新搜索模块的输出，所以需要对应数量的 gr.update()
    return (_update_status_bar(status_msg), *search_module_updates[1:]) # 第一个是查询框，不在此更新

def handle_remove_filter_tag(
    report_state: UIAssessmentReport, # pylint: disable=unused-argument
    tag_to_remove: str,
    search_query_text: str
) -> Tuple[gr.update, gr.update, gr.update]: # pylint: disable=no-member
    """移除过滤标签。"""
    global _current_search_filter_tags # pylint: disable=global-statement
    status_msg = ""
    if tag_to_remove and tag_to_remove.strip():
        tag = tag_to_remove.strip()
        if tag in _current_search_filter_tags:
            _current_search_filter_tags.remove(tag)
            status_msg = f"标签 '{tag}' 已从过滤器中移除。"
        else:
            status_msg = f"标签 '{tag}' 不在过滤器中。"
    else:
        status_msg = "请输入有效的标签名。"

    search_module_updates = update_evidence_search_module_components(
        current_search_query=search_query_text,
        current_filter_tags=_current_search_filter_tags,
        recommended_tags=_last_recommended_tags,
        search_results=_last_search_results
    )
    return (_update_status_bar(status_msg), *search_module_updates[1:])


def handle_search_evidence_click(
    report_state: UIAssessmentReport, # pylint: disable=unused-argument
    search_query: str,
    # navi_search_client: NaviSearchClient # 应该从 main.py 传入
    # 临时方案：在 main.py 中用 functools.partial 包装此回调以传入客户端
    # 或者，如果 NaviSearchClient 是无状态的，可以在这里实例化 (不推荐)
) -> Tuple[gr.update, gr.update, gr.update, gr.update, gr.update]: # pylint: disable=no-member
    """处理“执行搜索”按钮点击。"""
    global _last_search_results, _last_recommended_tags, _current_search_filter_tags # pylint: disable=global-statement

    # 实际的 navi_search_client 应该在 main.py 中实例化并传递
    # 这里我们模拟一个，或者期望它通过某种方式被注入
    # 例如： navi_search_client = gr.State(...) 然后作为输入参数
    # 为了能运行，暂时假设有一个可用的 mock 或实际的 client

    # ---- 获取 NaviSearchClient 实例 ----
    # 这部分逻辑需要在 main.py 中正确设置。
    # 假设 main.py 中有一个 get_navi_search_client() 函数
    # 或者 navi_search_client 是通过 gr.State 传递的。
    # 这里我们先留空，表示需要外部提供。
    navi_search_client_instance: Optional[NaviSearchClient] = getattr(handle_search_evidence_click, 'navi_search_client', None)
    if not navi_search_client_instance:
        _last_search_results = []
        _last_recommended_tags = []
        # pylint: disable=line-too-long
        search_updates = update_evidence_search_module_components(current_search_query=search_query, current_filter_tags=_current_search_filter_tags, recommended_tags=[], search_results=[])
        return _update_status_bar("NaviSearchClient 未初始化，无法执行搜索。", is_error=True), *search_updates[1:]


    status_msg = f"正在为查询 '{search_query}' (标签: {_current_search_filter_tags}) 搜索证据..."

    # 构造 AssessmentSpecItem (如果 NaviSearchClient 需要)
    # NaviSearchClient.search_evidence(spec_item: AssessmentSpecItem)
    # 我们需要一个 spec_item。如果搜索是通用的，而不是针对特定 spec_item，
    # NaviSearchClient 可能需要一个不同的接口或适配。
    # 假设我们基于当前UI的 spec_id 和 spec_content 来构建一个临时的 CoreAssessmentSpecItem
    current_ui_item = _get_current_item_from_state(report_state)
    if not current_ui_item:
        _last_search_results = []
        _last_recommended_tags = []
        search_updates = update_evidence_search_module_components(current_search_query=search_query, current_filter_tags=_current_search_filter_tags, recommended_tags=[], search_results=[])
        return _update_status_bar("没有当前评估项上下文，无法精确搜索。", is_error=True), *search_updates[1:]

    # 构造一个 CoreAssessmentSpecItem 供搜索使用
    # 注意：NaviSearchClient 的 search_evidence 可能期望一个完整的 spec_item
    # 或者它有一个更简单的搜索接口。这里我们假设它需要 spec_item。
    # 它的 search_evidence 返回 (List[CoreEvidenceSearchResult], EvidenceSearchParams)
    # 这里的 query_text 和 filter_tags 应该由 LLMClient.generate_search_params 生成，
    # 然后 NaviSearchClient._search_via_visitor_api 使用它们。
    # 为了简化UI的直接搜索，我们可能需要一个直接的搜索函数，或者模拟LLM的第一步。

    # 简化：直接使用用户输入的 query 和 tags。这可能与 NaviSearchClient 的迭代逻辑不完全匹配。
    # NaviSearchClient.search_evidence() 内部有迭代逻辑。
    # 我们需要一个简化的 spec_item 来启动这个过程。
    mock_spec_for_search = CoreAssessmentSpecItem(
        id=current_ui_item.spec_id,
        content=search_query, # 使用用户输入的查询作为内容驱动搜索
        heading="", # 可选
        condition="", # 可选
        method="" # 可选
    )

    try:
        # 搜索结果是 (List[CoreEvidenceSearchResult], EvidenceSearchParams)
        core_search_results, final_search_params = navi_search_client_instance.search_evidence(mock_spec_for_search)

        _last_search_results = [UIEvidence.from_core_search_result(res) for res in core_search_results]
        _last_recommended_tags = final_search_params.filter_tags # 或者从搜索结果中提取推荐标签

        status_msg = f"找到 {len(_last_search_results)} 条证据。"
        if not _last_search_results:
            status_msg += " 未找到匹配的证据。"

    except Exception as e: # pylint: disable=broad-except
        print(f"搜索证据时发生错误: {e}\n{traceback.format_exc()}")
        status_msg = f"搜索证据时发生错误: {e}"
        _last_search_results = []
        _last_recommended_tags = []

    # pylint: disable=line-too-long
    search_module_updates = update_evidence_search_module_components(
        current_search_query=search_query, # 用户输入的查询
        current_filter_tags=_current_search_filter_tags, # 当前过滤器中的标签
        recommended_tags=_last_recommended_tags, # LLM 或搜索引擎推荐的标签
        search_results=_last_search_results      # 搜索到的证据
    )
    return _update_status_bar(status_msg), *search_module_updates[1:]


def handle_reference_searched_evidence(
    report_state: UIAssessmentReport,
    selected_evidence_id: str # 来自搜索结果的 Radio 选择器
) -> Tuple[UIAssessmentReport, gr.update, gr.update, gr.update]: # pylint: disable=no-member
    """将选中的搜索结果证据添加到当前评估项的引用列表。"""
    current_item = _get_current_item_from_state(report_state)
    status_msg = "操作失败。"
    ref_ev_display_html_update = gr.update() # type: ignore
    ref_ev_selector_update = gr.update() # type: ignore
    ref_ev_actions_row_update = gr.update() # type: ignore

    if current_item and selected_evidence_id:
        evidence_to_add = next((ev for ev in _last_search_results if ev.evidence_id == selected_evidence_id), None)

        if evidence_to_add:
            # 检查是否已存在 (基于 evidence_id)
            if any(ref_ev.evidence_id == evidence_to_add.evidence_id for ref_ev in current_item.referenced_evidences):
                status_msg = f"证据 '{selected_evidence_id}' 已存在于引用列表中。"
            else:
                # 创建 UIEvidence 的新实例副本以添加到引用列表
                new_ref_evidence = evidence_to_add.model_copy(deep=True)
                new_ref_evidence.is_active_for_conclusion = True # 默认激活
                current_item.referenced_evidences.append(new_ref_evidence)
                status_msg = f"证据 '{selected_evidence_id}' 已添加至引用列表。"
        else:
            status_msg = f"未在上次搜索结果中找到证据ID: {selected_evidence_id}"

        # 更新引用证据区的显示
        html_parts = [format_evidence_to_markdown(ev, is_referenced=True, index=i)
                      for i, ev in enumerate(current_item.referenced_evidences)]
        ref_ev_display_html_update = gr.update(value="".join(html_parts) if html_parts else "无引用证据") # type: ignore

        new_choices = [ev.evidence_id for ev in current_item.referenced_evidences]
        ref_ev_selector_update = gr.update(choices=new_choices, value=None, visible=bool(new_choices)) # type: ignore
        ref_ev_actions_row_update = gr.update(visible=bool(new_choices)) # type: ignore

    elif not current_item:
        status_msg = "没有当前评估项来引用证据。"
    elif not selected_evidence_id:
        status_msg = "请先选择一个搜索到的证据进行引用。"

    return report_state, _update_status_bar(status_msg), ref_ev_display_html_update, ref_ev_selector_update, ref_ev_actions_row_update


# --- LLM 交互回调 ---
def handle_regenerate_conclusion_click(
    report_state: UIAssessmentReport,
    # llm_client: LLMAssessmentClient # 应该从 main.py 传入
) -> Tuple[UIAssessmentReport, gr.update, gr.update, gr.update, gr.update, gr.update, gr.update]: # pylint: disable=no-member
    """处理“重新生成结论”按钮点击。"""
    current_item = _get_current_item_from_state(report_state)
    status_msg = "操作失败。"
    # LLM 对比区域的更新
    llm_area_visible_update = gr.update(visible=False) # type: ignore
    llm_curr_j_update = gr.update() # type: ignore
    llm_curr_c_update = gr.update() # type: ignore
    llm_sugg_j_update = gr.update() # type: ignore
    llm_sugg_c_update = gr.update() # type: ignore

    if not current_item:
        status_msg = "没有当前评估项，无法为其生成结论。"
        return report_state, _update_status_bar(status_msg, True), llm_area_visible_update, llm_curr_j_update, llm_curr_c_update, llm_sugg_j_update, llm_sugg_c_update

    # ---- 获取 LLMClient 实例 ----
    llm_client_instance: Optional[LLMAssessmentClient] = getattr(handle_regenerate_conclusion_click, 'llm_client', None)
    if not llm_client_instance:
        return report_state, _update_status_bar("LLMClient 未初始化，无法生成结论。", is_error=True), llm_area_visible_update, llm_curr_j_update, llm_curr_c_update, llm_sugg_j_update, llm_sugg_c_update


    status_msg = f"正在为评估项 '{current_item.spec_id}' 调用LLM生成建议结论..."

    # 准备LLM输入
    # 1. 评估规范条目
    core_spec_item = CoreAssessmentSpecItem(
        id=current_item.spec_id,
        content=current_item.spec_content,
        heading="", condition="", method="" # 根据LLMClient的需要填充
    )
    # 2. 当前“引用证据区”中所有状态为“已激活”的证据
    active_evidences_for_llm: List[CoreEvidenceSearchResult] = []
    for ui_ev in current_item.referenced_evidences:
        if ui_ev.is_active_for_conclusion:
            active_evidences_for_llm.append(
                CoreEvidenceSearchResult(source=ui_ev.evidence_id, content=ui_ev.content)
            )

    try:
        # 调用LLM
        llm_conclusion: CoreConclusion = llm_client_instance.generate_assessment(core_spec_item, active_evidences_for_llm)

        # 更新当前项的LLM建议字段
        current_item.llm_suggested_judgement = Judgement(llm_conclusion.judgement.value) #确保类型正确
        current_item.llm_suggested_comment = llm_conclusion.comment

        # 更新LLM建议对比区
        llm_area_visible_update = gr.update(visible=True) # type: ignore
        llm_curr_j_update = gr.update(value=current_item.current_judgement.value) # type: ignore
        llm_curr_c_update = gr.update(value=current_item.current_comment) # type: ignore
        llm_sugg_j_update = gr.update(value=current_item.llm_suggested_judgement.value) # type: ignore
        llm_sugg_c_update = gr.update(value=current_item.llm_suggested_comment) # type: ignore

        status_msg = "LLM建议已生成并显示在对比区。"

    except Exception as e: # pylint: disable=broad-except
        print(f"LLM调用失败: {e}\n{traceback.format_exc()}")
        status_msg = f"LLM调用失败: {e}"
        current_item.llm_suggested_judgement = None
        current_item.llm_suggested_comment = None
        # 保持对比区隐藏或显示错误
        llm_area_visible_update = gr.update(visible=False) # type: ignore
        # 清空对比区内容
        llm_curr_j_update = gr.update(value="") # type: ignore
        llm_curr_c_update = gr.update(value="") # type: ignore
        llm_sugg_j_update = gr.update(value="") # type: ignore
        llm_sugg_c_update = gr.update(value="") # type: ignore

    return report_state, _update_status_bar(status_msg), llm_area_visible_update, llm_curr_j_update, llm_curr_c_update, llm_sugg_j_update, llm_sugg_c_update


def handle_adopt_llm_suggestion(
    report_state: UIAssessmentReport
) -> Tuple[UIAssessmentReport, gr.update, gr.update, gr.update, gr.update, gr.update]: # pylint: disable=no-member
    """处理“采纳LLM建议”按钮点击。"""
    current_item = _get_current_item_from_state(report_state)
    status_msg = "操作失败。"
    # UI 更新：判断、评论、引用证据区、LLM对比区可见性
    judgement_update = gr.update() # type: ignore
    comment_update = gr.update() # type: ignore
    ref_ev_display_update = gr.update() # type: ignore
    llm_area_visible_update = gr.update(visible=False) # type: ignore

    if current_item and current_item.llm_suggested_judgement is not None:
        # 更新结论和评论
        current_item.current_judgement = current_item.llm_suggested_judgement
        current_item.current_comment = current_item.llm_suggested_comment or ""

        # 移除被抑制的证据
        current_item.referenced_evidences = [
            ev for ev in current_item.referenced_evidences if ev.is_active_for_conclusion
        ]

        # 清空LLM建议字段
        current_item.llm_suggested_judgement = None
        current_item.llm_suggested_comment = None

        # 更新UI
        judgement_update = gr.update(value=current_item.current_judgement.value) # type: ignore
        comment_update = gr.update(value=current_item.current_comment) # type: ignore

        html_parts = [format_evidence_to_markdown(ev, is_referenced=True, index=i)
                      for i, ev in enumerate(current_item.referenced_evidences)]
        ref_ev_display_update = gr.update(value="".join(html_parts) if html_parts else "无引用证据") # type: ignore

        # 更新审核状态 (如果之前未审核)
        if current_item.audit_status == ItemAuditStatus.NOT_REVIEWED and \
           current_item.current_judgement != Judgement.NOT_PROCESSED:
            current_item.audit_status = ItemAuditStatus.REVIEWED
        report_state.update_stats()
        stats_updates = update_statistics_components(report_state) # type: ignore

        status_msg = "已采纳LLM建议。被抑制的证据已移除。"
        # 返回 (report_state, status_bar, judgement_dd, comment_tb, ref_ev_display, llm_area_visibility, stat_judgement_md, stat_audit_md)
        # pylint: disable=line-too-long
        return report_state, _update_status_bar(status_msg), judgement_update, comment_update, ref_ev_display_update, llm_area_visible_update, stats_updates[0], stats_updates[1]

    elif not current_item:
        status_msg = "没有当前评估项。"
    else: # 没有LLM建议
        status_msg = "没有可采纳的LLM建议。"

    # pylint: disable=line-too-long
    return report_state, _update_status_bar(status_msg, is_error=not (current_item and current_item.llm_suggested_judgement is not None)), judgement_update, comment_update, ref_ev_display_update, llm_area_visible_update, gr.update(), gr.update()


def handle_discard_llm_suggestion(
    report_state: UIAssessmentReport
) -> Tuple[UIAssessmentReport, gr.update, gr.update]: # pylint: disable=no-member
    """处理“放弃/保留我的修改”按钮点击。"""
    current_item = _get_current_item_from_state(report_state)
    status_msg = "操作完成。"

    if current_item:
        # 清空LLM建议字段，保持用户当前结论不变
        current_item.llm_suggested_judgement = None
        current_item.llm_suggested_comment = None
        status_msg = "LLM建议已忽略，保留当前修改。"
    else:
        status_msg = "没有当前评估项。"

    llm_area_visible_update = gr.update(visible=False) # type: ignore
    return report_state, _update_status_bar(status_msg), llm_area_visible_update

