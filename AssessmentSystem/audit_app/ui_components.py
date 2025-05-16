# audit_app/ui_components.py
import gradio as gr # type: ignore # pylint: disable=import-error
from typing import List, Optional, Callable, Dict, Tuple, Any

# 导入UI模型和核心模型
# 使用相对项目根目录的导入方式
from AssessmentSystem.audit_app.models_ui import (
    UIAssessmentItem,
    UIEvidence,
    Judgement,
    ItemAuditStatus,
    EvidenceSearchParams,
    UIAssessmentReport
)
# 尝试导入核心Python类，用于类型提示和可能的直接调用
try:
    from AssessmentSystem.model import AssessmentSpecItem as CoreAssessmentSpecItem # pylint: disable=import-error
    from AssessmentSystem.llm_client import LLMAssessmentClient # pylint: disable=import-error
    from AssessmentSystem.navi_search_client import NaviSearchClient # pylint: disable=import-error
except ImportError:
    print("警告: AssessmentSystem 中的一个或多个核心模块无法导入。某些类型提示可能不准确，且与后端类的直接交互将失败。")
    LLMAssessmentClient = None # type: ignore
    NaviSearchClient = None # type: ignore
    CoreAssessmentSpecItem = None # type: ignore


# --- 辅助函数 ---
def format_evidence_to_markdown(evidence: UIEvidence, is_referenced: bool, index: Optional[int] = None, is_search_result: bool = False) -> str:
    """
    将单个 UIEvidence 对象格式化为 Markdown 字符串，用于在Gradio中显示。
    包含激活/抑制按钮（如果是引用证据）或引用按钮（如果是搜索结果）。

    参数:
        evidence (UIEvidence): 要格式化的证据对象。
        is_referenced (bool): 如果为True，则此证据是当前评估项引用的证据。
        index (Optional[int]): 如果是引用证据，则为其在列表中的索引（0-based）。
        is_search_result (bool): 如果为True，则此证据是搜索结果列表中的一项。

    返回:
        str: Markdown 格式的证据卡片。
    """
    card_md = f"**ID:** `{evidence.evidence_id}`\n"
    if evidence.title:
        card_md += f"**标题:** {evidence.title}\n"
    if evidence.evidence_type:
        card_md += f"**类型:** {evidence.evidence_type}\n"
    if evidence.timestamp_str:
        card_md += f"**时间戳:** {evidence.timestamp_str}\n"

    card_md += f"**内容预览:**\n```text\n{evidence.short_content}\n```\n"

    if evidence.search_tags:
        tags_str = ", ".join([f"`{tag}`" for tag in evidence.search_tags])
        card_md += f"**相关标签:** {tags_str}\n"

    # 根据上下文添加按钮的占位符或实际的交互提示
    # 实际的按钮是Gradio组件，这里的Markdown只是视觉表示，交互通过外部按钮处理
    if is_referenced:
        status_text = "🟢 已激活" if evidence.is_active_for_conclusion else "⚪️ 已抑制 (待移除)"
        card_md += f"**状态:** {status_text}\n"
        # 按钮的标签将用于回调函数的 gr.Button().click(..., inputs=[gr.Textbox(value=evidence.evidence_id, visible=False), ...])
        # 这里只是示意，实际按钮在外部创建
        # card_md += f"\n<small>操作: [切换状态] [移除]</small>\n" # 这种方式在纯Markdown中无法直接交互
    elif is_search_result:
        # card_md += f"\n<small>操作: [引用此证据]</small>\n"
        pass # “引用此证据”按钮将在外部创建

    style = ""
    if is_referenced and not evidence.is_active_for_conclusion:
        style = "opacity: 0.6; border-left: 3px solid gray;" # 抑制的证据样式

    return f"<div style='border: 1px solid #e0e0e0; padding: 10px; margin-bottom: 10px; border-radius: 5px; {style}'>{card_md}</div>"

def create_tag_buttons_layout(
    tags: List[str],
    on_click_callback: Callable[..., Any], # pylint: disable=unused-argument
    tag_type: str, # "filter" 或 "ranked" # pylint: disable=unused-argument
    interactive: bool = True
) -> List[gr.components.Button]: # pylint: disable=no-member
    """
    为给定的标签列表创建一行Gradio按钮。
    点击按钮会调用提供的回调函数。

    参数:
        tags (List[str]): 要为其创建按钮的标签字符串列表。
        on_click_callback (Callable): 点击标签按钮时调用的函数。
                                      它应该接受标签值作为参数。
        tag_type (str): 用于区分回调函数中的标签来源 ("filter" 或 "ranked")。
        interactive (bool): 按钮是否可交互。

    返回:
        List[gr.Button]: 创建的Gradio按钮列表。
    """
    buttons: List[gr.components.Button] = [] # pylint: disable=no-member
    if not tags:
        # mypy: Argument 1 to "Markdown" has incompatible type "str"; expected "Optional[Callable[[], Any]]"
        # Gradio 的 Markdown 组件期望一个值或一个返回值的函数
        return [gr.Markdown(value="无")] # type: ignore # pylint: disable=no-member
    # pylint: disable=no-member
    # type: ignore
    with gr.Row(wrap=True):
        for tag_text in tags:
            # 将标签值和类型作为隐藏输入传递给回调
            # Gradio 的 Button.click() 的 inputs 参数可以直接是组件，也可以是 Python 值
            # 为了清晰，我们这里不直接用 gr.Textbox(value=tag_text, visible=False)
            # 而是在回调函数中处理。回调函数需要知道哪个按钮被点击了。
            # 一个简单的方法是让回调函数接收按钮的标签值。
            # Gradio 的新版本允许按钮直接传递其值，或者我们可以用 functools.partial

            # 这里我们假设回调函数能够通过某种方式（例如，部分应用或闭包）知道被点击的标签
            # 或者，更简单的是，按钮的 .click() 方法可以传递按钮的 value
            btn = gr.Button(value=tag_text, size="sm", interactive=interactive) # type: ignore # pylint: disable=no-member
            # btn.click(lambda t=tag_text: on_click_callback(t, tag_type), inputs=None, outputs=None) # 这种方式在Gradio中可能需要更复杂的处理
            # 更常见的方式是，回调函数会更新一个状态，然后重新渲染UI
            buttons.append(btn)
    return buttons


# --- UI 模块创建函数 ---

def create_top_nav_bar() -> Dict[str, gr.components.Button | gr.components.Markdown]: # pylint: disable=no-member
    """创建顶部导航栏和控制按钮。"""
    with gr.Row(equal_height=False): # type: ignore # pylint: disable=no-member
        prev_btn = gr.Button("⬅️ 上一条", variant="secondary") # type: ignore # pylint: disable=no-member
        skip_btn = gr.Button("➡️ 跳过本条", variant="secondary") # type: ignore # pylint: disable=no-member
        next_btn = gr.Button("✅ 通过并下一条", variant="primary") # type: ignore # pylint: disable=no-member
        # 重新生成结论按钮
        regenerate_conclusion_btn = gr.Button("🔄 重新生成结论 (LLM)", variant="secondary") # type: ignore # pylint: disable=no-member
        save_work_btn = gr.Button("💾 保存当前工作", variant="secondary") # type: ignore # pylint: disable=no-member

    progress_display = gr.Markdown("正在审核: - / - (ID: -)") # type: ignore # pylint: disable=no-member

    return {
        "prev_btn": prev_btn,
        "skip_btn": skip_btn,
        "next_btn": next_btn,
        "regenerate_btn": regenerate_conclusion_btn,
        "save_btn": save_work_btn,
        "progress_display": progress_display
    }

def create_report_detail_module(report_data: Optional[UIAssessmentReport] = None) -> Dict[str, Any]:
    """
    创建报告详情与编辑模块 (左栏)。
    包含评估规范展示区、评估结论编辑区、引用证据区。
    """
    current_item = report_data.get_current_assessment_item() if report_data else None

    # 1. 评估规范展示区 (StandardDisplayArea)
    # Gradio 的 Blocks, Column, Row 等布局元素在 Pylint 中可能被认为是动态的或类型不明确
    with gr.Blocks(): # type: ignore # pylint: disable=no-member
        gr.Markdown("### 1. 评估规范详情") # type: ignore # pylint: disable=no-member
        spec_id_display = gr.Textbox( # type: ignore # pylint: disable=no-member
            label="评估项 ID",
            value=current_item.spec_id if current_item else "N/A",
            interactive=False
        )
        spec_content_display = gr.Textbox( # type: ignore # pylint: disable=no-member
            label="评估规范内容",
            value=current_item.spec_content if current_item else "N/A",
            lines=5,
            interactive=False,
            show_copy_button=True
        )

        # 2. 评估结论编辑区 (ConclusionEditArea)
        gr.Markdown("### 2. 评估结论编辑") # type: ignore # pylint: disable=no-member
        judgement_options = [j.value for j in Judgement if j != Judgement.ERROR] # 不包括ERROR作为可选
        current_judgement_dd = gr.Dropdown( # type: ignore # pylint: disable=no-member
            label="判断结论 (Judgement)",
            choices=judgement_options,
            value=current_item.current_judgement.value if current_item else Judgement.NOT_PROCESSED.value,
            interactive=True
        )
        current_comment_tb = gr.Textbox( # type: ignore # pylint: disable=no-member
            label="评审意见/备注 (Comment)",
            value=current_item.current_comment if current_item else "",
            lines=5,
            interactive=True,
            placeholder="请输入您的评审意见或备注..."
        )

        # LLM建议对比区 (LLMComparisonArea) - 默认隐藏，条件触发显示
        with gr.Column(visible=False) as llm_comparison_area: # type: ignore # pylint: disable=no-member
            gr.Markdown("#### LLM 建议对比") # type: ignore # pylint: disable=no-member
            with gr.Row(): # type: ignore # pylint: disable=no-member
                with gr.Column(scale=1): # type: ignore # pylint: disable=no-member
                    gr.Markdown("**您的当前结论**") # type: ignore # pylint: disable=no-member
                    llm_compare_current_judgement = gr.Textbox(label="当前判断", interactive=False) # type: ignore # pylint: disable=no-member
                    llm_compare_current_comment = gr.Textbox(label="当前备注", lines=3, interactive=False) # type: ignore # pylint: disable=no-member
                with gr.Column(scale=1): # type: ignore # pylint: disable=no-member
                    gr.Markdown("**LLM 建议结论**") # type: ignore # pylint: disable=no-member
                    llm_compare_suggested_judgement = gr.Textbox(label="LLM建议判断", interactive=False) # type: ignore # pylint: disable=no-member
                    llm_compare_suggested_comment = gr.Textbox(label="LLM建议备注", lines=3, interactive=False) # type: ignore # pylint: disable=no-member
            with gr.Row(): # type: ignore # pylint: disable=no-member
                adopt_llm_btn = gr.Button("采纳LLM建议", variant="primary") # type: ignore # pylint: disable=no-member
                discard_llm_btn = gr.Button("放弃/保留我的修改", variant="secondary") # type: ignore # pylint: disable=no-member

        # 3. 引用证据区 (ReferencedEvidenceArea)
        gr.Markdown("### 3. 引用证据列表") # type: ignore # pylint: disable=no-member
        referenced_evidences_display = gr.HTML("无引用证据") # type: ignore # pylint: disable=no-member # 使用HTML组件以支持更丰富的格式

        referenced_evidence_selector = gr.Radio( # type: ignore # pylint: disable=no-member
            label="选择要操作的引用证据 (通过ID)",
            choices=[], # 将由回调填充
            value=None,
            interactive=True,
            visible=False # 初始隐藏，有证据时显示
        )
        with gr.Row(visible=False) as ref_evidence_actions_row: # type: ignore # pylint: disable=no-member
            toggle_evidence_btn = gr.Button("切换选中证据的激活/抑制状态", size="sm") # type: ignore # pylint: disable=no-member
            remove_evidence_btn = gr.Button("移除选中证据", variant="stop", size="sm") # type: ignore # pylint: disable=no-member

    return { # pylint: disable=possibly-used-before-assignment
        "spec_id": spec_id_display,
        "spec_content": spec_content_display,
        "judgement": current_judgement_dd,
        "comment": current_comment_tb,
        "llm_comparison_area": llm_comparison_area,
        "llm_compare_current_judgement": llm_compare_current_judgement,
        "llm_compare_current_comment": llm_compare_current_comment,
        "llm_compare_suggested_judgement": llm_compare_suggested_judgement,
        "llm_compare_suggested_comment": llm_compare_suggested_comment,
        "adopt_llm_btn": adopt_llm_btn,
        "discard_llm_btn": discard_llm_btn,
        "referenced_evidences_display": referenced_evidences_display,
        "referenced_evidence_selector": referenced_evidence_selector,
        "ref_evidence_actions_row": ref_evidence_actions_row,
        "toggle_evidence_btn": toggle_evidence_btn,
        "remove_evidence_btn": remove_evidence_btn,
    }

def create_evidence_search_module() -> Dict[str, Any]:
    """
    创建证据搜索与浏览模块 (右栏)。
    包含搜索参数编辑区、搜索结果展示区。
    """
    with gr.Blocks(): # type: ignore # pylint: disable=no-member
        gr.Markdown("### 4. 证据搜索与浏览") # type: ignore # pylint: disable=no-member

        # 1. 搜索参数编辑区 (SearchParametersArea)
        gr.Markdown("#### 搜索参数") # type: ignore # pylint: disable=no-member
        search_query_tb = gr.Textbox(label="查询文本 (Query Text)", placeholder="输入搜索关键词...", interactive=True) # type: ignore # pylint: disable=no-member

        gr.Markdown("已选过滤标签 (Filter Tags):") # type: ignore # pylint: disable=no-member
        selected_filter_tags_container = gr.HTML("无") # type: ignore # pylint: disable=no-member # 将由回调填充

        filter_tag_input_tb = gr.Textbox(label="添加/移除过滤标签", placeholder="输入标签名按回车或点击按钮", interactive=True) # type: ignore # pylint: disable=no-member
        with gr.Row(): # type: ignore # pylint: disable=no-member
            add_filter_tag_btn = gr.Button("添加标签", size="sm") # type: ignore # pylint: disable=no-member
            remove_filter_tag_btn = gr.Button("移除标签", size="sm") # type: ignore # pylint: disable=no-member

        search_btn = gr.Button("🔍 执行搜索", variant="primary") # type: ignore # pylint: disable=no-member

        with gr.Accordion("高级搜索选项", open=False): # type: ignore # pylint: disable=no-member
            gr.Markdown("高级搜索选项待实现 (如数量上限, 时间范围等)。") # type: ignore # pylint: disable=no-member

        # 2. 搜索结果展示区 (SearchResultsArea)
        gr.Markdown("#### 搜索结果") # type: ignore # pylint: disable=no-member
        gr.Markdown("推荐过滤标签 (Ranked Tags):") # type: ignore # pylint: disable=no-member
        recommended_tags_container = gr.HTML("无") # type: ignore # pylint: disable=no-member # 将由回调填充

        gr.Markdown("排序后的证据记录 (Ranked Records):") # type: ignore # pylint: disable=no-member
        search_results_display = gr.HTML("执行搜索以查看结果。") # type: ignore # pylint: disable=no-member # 用于显示证据卡片列表

        search_result_selector = gr.Radio( # type: ignore # pylint: disable=no-member
            label="选择要引用的搜索结果 (通过ID)",
            choices=[],
            value=None,
            interactive=True,
            visible=False # 初始隐藏
        )
        with gr.Row(visible=False) as search_result_actions_row: # type: ignore # pylint: disable=no-member
            reference_searched_evidence_btn = gr.Button("引用选中证据", size="sm") # type: ignore # pylint: disable=no-member

    return { # pylint: disable=possibly-used-before-assignment
        "search_query_tb": search_query_tb,
        "selected_filter_tags_container": selected_filter_tags_container,
        "filter_tag_input_tb": filter_tag_input_tb,
        "add_filter_tag_btn": add_filter_tag_btn,
        "remove_filter_tag_btn": remove_filter_tag_btn,
        "search_btn": search_btn,
        "recommended_tags_container": recommended_tags_container,
        "search_results_display": search_results_display,
        "search_result_selector": search_result_selector,
        "search_result_actions_row": search_result_actions_row,
        "reference_searched_evidence_btn": reference_searched_evidence_btn,
    }

def create_statistics_display(report_data: Optional[UIAssessmentReport] = None) -> Dict[str, gr.components.Markdown]: # pylint: disable=no-member
    """创建统计信息显示区域。"""

    initial_judgement_stats_md = "无数据"
    initial_audit_status_stats_md = "无数据"

    if report_data:
        # 判断结论统计
        judgement_values = [j.value for j in Judgement]
        judgement_counts = [report_data.stats_judgement.get(j_val, 0) for j_val in judgement_values]

        md_parts = ["**按判断结论统计:**"]
        for j_val, count in zip(judgement_values, judgement_counts):
            md_parts.append(f"- {j_val}: {count}")
        initial_judgement_stats_md = "\n".join(md_parts)

        # 审核状态统计
        audit_status_values = [s.value for s in ItemAuditStatus]
        audit_status_counts = [report_data.stats_audit_status.get(s_val, 0) for s_val in audit_status_values]

        md_parts = ["**按审核状态统计:**"]
        for s_val, count in zip(audit_status_values, audit_status_counts):
            md_parts.append(f"- {s_val}: {count}")
        initial_audit_status_stats_md = "\n".join(md_parts)

    with gr.Accordion("报告统计数据", open=True): # type: ignore # pylint: disable=no-member
        judgement_stats_md = gr.Markdown(initial_judgement_stats_md) # type: ignore # pylint: disable=no-member
        audit_status_stats_md = gr.Markdown(initial_audit_status_stats_md) # type: ignore # pylint: disable=no-member

    return {
        "judgement_stats_md": judgement_stats_md,
        "audit_status_stats_md": audit_status_stats_md
    }

def create_status_bar() -> gr.components.Textbox: # pylint: disable=no-member
    """创建底部状态栏。"""
    status_bar = gr.Textbox( # type: ignore # pylint: disable=no-member
        label="状态/日志",
        value="应用程序已启动。请加载报告开始审核。",
        lines=1,
        interactive=False,
        show_label=False,
        placeholder="最后操作的状态将显示在这里..."
    )
    return status_bar

def update_report_detail_module_components( # pylint: disable=too-many-locals
    components: Dict[str, Any],
    item: Optional[UIAssessmentItem],
    report_total_items: int,
    current_idx: int
    ) -> List[Any]:
    """
    辅助函数，用于更新左侧报告详情模块的所有组件的值。
    返回一个与 components 字典的 gr.update() 调用顺序相匹配的值列表。
    Pylint: disable=too-many-locals due to the number of UI elements being updated.
    """
    # pylint: disable=line-too-long
    # The return list matches the order of outputs expected by the callback.
    # Each gr.update(...) is an item in this list.
    # Note: gr.update() is the modern way to update components.
    # For older Gradio versions, one might return direct values or dictionaries.
    # We will return dictionaries that gr.update can process.

    if item:
        # 更新进度显示 - 这个通常是独立的组件，在主回调中单独处理
        # progress_text = f"正在审核: {current_idx + 1} / {report_total_items} (ID: {item.spec_id}, 状态: {item.audit_status.value})"

        ref_ev_html_parts = []
        ref_ev_ids_for_selector = []
        if item.referenced_evidences:
            for i, ev in enumerate(item.referenced_evidences):
                ref_ev_html_parts.append(format_evidence_to_markdown(ev, is_referenced=True, index=i))
                ref_ev_ids_for_selector.append(ev.evidence_id)
            ref_ev_display_html = "".join(ref_ev_html_parts) if ref_ev_html_parts else "无引用证据"
            ref_ev_selector_visible = True
        else:
            ref_ev_display_html = "无引用证据"
            ref_ev_selector_visible = False

        # 获取llm_comparison_area的当前可见性，因为我们不在此函数中改变它，除非特定逻辑需要
        # llm_area_visibility = components["llm_comparison_area"].visible # This would require passing the component itself
        # For simplicity, assume it's handled by a dedicated callback or its visibility is part of the UIAssessmentItem state.
        # Here, we'll just pass a gr.update() for its parts if they need resetting or based on some logic.
        # If llm_comparison_area visibility is managed by a gr.State or other callbacks, this might be simpler.
        # For now, let's assume we only update its content if item.llm_suggested_judgement is present.

        llm_area_visible_update = gr.update(visible=bool(item.llm_suggested_judgement is not None)) # type: ignore
        llm_current_j_update = gr.update(value=item.current_judgement.value if item.llm_suggested_judgement else "") # type: ignore
        llm_current_c_update = gr.update(value=item.current_comment if item.llm_suggested_judgement else "") # type: ignore
        llm_suggested_j_update = gr.update(value=item.llm_suggested_judgement.value if item.llm_suggested_judgement else "") # type: ignore
        llm_suggested_c_update = gr.update(value=item.llm_suggested_comment if item.llm_suggested_comment else "") # type: ignore


        return [
            gr.update(value=item.spec_id), # spec_id
            gr.update(value=item.spec_content), # spec_content
            gr.update(value=item.current_judgement.value), # judgement
            gr.update(value=item.current_comment), # comment
            gr.update(value=ref_ev_display_html), # referenced_evidences_display
            gr.update(choices=ref_ev_ids_for_selector, value=None, visible=ref_ev_selector_visible), # referenced_evidence_selector
            gr.update(visible=ref_ev_selector_visible), # ref_evidence_actions_row
            llm_area_visible_update, # llm_comparison_area (Column)
            llm_current_j_update, # llm_compare_current_judgement
            llm_current_c_update, # llm_compare_current_comment
            llm_suggested_j_update, # llm_compare_suggested_judgement
            llm_suggested_c_update, # llm_compare_suggested_comment
        ]
    else:
        # progress_text = "无评估项可审核。"
        return [
            gr.update(value="N/A"),
            gr.update(value="N/A"),
            gr.update(value=Judgement.NOT_PROCESSED.value),
            gr.update(value=""),
            gr.update(value="无引用证据"),
            gr.update(choices=[], value=None, visible=False),
            gr.update(visible=False),
            gr.update(visible=False), # llm_comparison_area
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
        ]

def update_evidence_search_module_components( # pylint: disable=too-many-locals
    # components: Dict[str, Any], # Not strictly needed if we return direct gr.update calls
    current_search_query: Optional[str] = None,
    current_filter_tags: Optional[List[str]] = None,
    recommended_tags: Optional[List[str]] = None, # Changed from List[UIEvidence] to List[str]
    search_results: Optional[List[UIEvidence]] = None
    ) -> List[Any]:
    """
    辅助函数，用于更新右侧证据搜索模块的组件值。
    """
    sel_filter_tags_html = "无"
    if current_filter_tags:
        sel_filter_tags_html = ", ".join([f"`{tag}`" for tag in current_filter_tags]) if current_filter_tags else "无"

    rec_tags_html = "无"
    if recommended_tags:
        rec_tags_html = ", ".join([f"`{tag}`" for tag in recommended_tags]) if recommended_tags else "无"

    search_res_html_parts = []
    search_res_ids_for_selector = []
    search_res_selector_visible = False
    if search_results:
        for i, ev in enumerate(search_results):
            search_res_html_parts.append(format_evidence_to_markdown(ev, is_referenced=False, is_search_result=True, index=i))
            search_res_ids_for_selector.append(ev.evidence_id)
        search_res_display_html = "".join(search_res_html_parts) if search_res_html_parts else "未找到匹配的证据。"
        if search_results: # Only show selector if there are results
             search_res_selector_visible = True
    else:
        search_res_display_html = "执行搜索以查看结果。"
        search_res_selector_visible = False


    return [
        gr.update(value=current_search_query or ""), # search_query_tb
        gr.update(value=sel_filter_tags_html),       # selected_filter_tags_container
        gr.update(value=rec_tags_html),              # recommended_tags_container
        gr.update(value=search_res_display_html),    # search_results_display
        gr.update(choices=search_res_ids_for_selector, value=None, visible=search_res_selector_visible), # search_result_selector
        gr.update(visible=search_res_selector_visible) # search_result_actions_row
    ]


def update_statistics_components(
    # components: Dict[str, gr.components.Markdown],  # Not strictly needed
    report_data: UIAssessmentReport
    ) -> List[Any]:
    """更新统计信息显示组件。"""
    report_data.update_stats()

    judgement_values = [j.value for j in Judgement]
    judgement_counts = [report_data.stats_judgement.get(j_val, 0) for j_val in judgement_values]

    jdg_md_parts = ["**按判断结论统计:**"]
    for j_val, count in zip(judgement_values, judgement_counts):
        jdg_md_parts.append(f"- {j_val}: {count}")
    judgement_stats_md_val = "\n".join(jdg_md_parts)

    audit_status_values = [s.value for s in ItemAuditStatus]
    audit_status_counts = [report_data.stats_audit_status.get(s_val, 0) for s_val in audit_status_values]

    aud_md_parts = ["**按审核状态统计:**"]
    for s_val, count in zip(audit_status_values, audit_status_counts):
        aud_md_parts.append(f"- {s_val}: {count}")
    audit_stats_md_val = "\n".join(aud_md_parts)

    return [
        gr.update(value=judgement_stats_md_val), # judgement_stats_md
        gr.update(value=audit_stats_md_val)      # audit_status_stats_md
    ]

