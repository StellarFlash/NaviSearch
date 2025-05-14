import os
import dotenv
import gradio as gr
import requests
import json

dotenv.load_dotenv() # 加载 .env 文件中的环境变量到 environmen

# FastAPI服务地址
VISITOR_API_URL = os.getenv("VISITOR_API_HOST") +":" + os.getenv("VISITOR_API_PORT")# 请确保您的FastAPI服务在此地址运行

# 初始默认的标签池 (如果需要在没有任何搜索结果时显示一些标签)
initial_filter_tags_pool = [
    "SMF", "双向认证", "3GPP TS 33.501", "网络拓扑图", "LST CPNODEID",
    "LST CPACCESSLISTFUNC", "身份认证与授权(Authentication & Authorization)",
    "IP配置互通", "GTP-U", "LST CPACCESSLIST"
]

# --- API 调用函数 ---
def call_search_api(query_text: str, active_tags_list: list, mode: str, retrieval_top_k: int, rerank_strategy: str, rerank_top_k_standard: int):
    """调用后端的Search接口"""
    payload = {
        "query_text": query_text,
        "filter_tags": active_tags_list, # API中的 filter_tags 对应我们前端的 active_tags
        "mode": mode,
        "retrieval_top_k": retrieval_top_k,
        "rerank_strategy": rerank_strategy,
        "rerank_top_k_standard": rerank_top_k_standard
    }
    try:
        response = requests.post(f"{VISITOR_API_URL}/search", json=payload)
        response.raise_for_status() # 如果HTTP请求返回了错误状态码，则抛出HTTPError异常
        return response.json()
    except requests.exceptions.RequestException as e:
        # 处理网络请求相关的错误 (e.g., 连接错误, 超时)
        return {"status": "error", "message": f"API请求失败: {e}", "ranked_records": [], "ranked_tags": []}
    except json.JSONDecodeError:
        # 处理API返回非JSON格式的错误
        return {"status": "error", "message": "API返回格式错误，无法解析JSON", "ranked_records": [], "ranked_tags": []}
    except Exception as e:
        # 处理其他未知错误
        return {"status": "error", "message": f"未知错误: {str(e)}", "ranked_records": [], "ranked_tags": []}


# --- Gradio 应用定义 ---
with gr.Blocks(title="NaviSearch 调试界面", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## NaviSearch 调试界面")

    # --- 状态变量 (State Variables) ---
    # s_activated_tags_list: 存储当前用户已激活的标签列表 (Python list of strings)
    s_activated_tags_list = gr.State([])
    # s_ranked_tags_from_api: 存储上一次搜索API返回的 ranked_tags 列表 (Python list of strings)
    s_ranked_tags_from_api = gr.State([])

    with gr.Row():
        # --- 左侧栏: 搜索参数和标签管理 ---
        with gr.Column(scale=1): # scale 控制列的相对宽度
            gr.Markdown("### 搜索参数")
            query_input = gr.Textbox(label="查询语句 (Query)", placeholder="输入查询内容...")
            mode_radio = gr.Radio(["standard", "agentic"], label="搜索模式 (Mode)", value="standard")
            retrieval_top_k_slider = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="召回数量 (Retrieval Top K)")
            rerank_strategy_dropdown = gr.Dropdown(["ranking", "filtering"], label="重排策略 (Rerank Strategy)", value="ranking")
            rerank_top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="重排结果数量 (Rerank Top K)")

            gr.Markdown("### 激活标签 (Activated Tags)")
            activated_tags_checkbox_group = gr.CheckboxGroup(
                label="当前激活的标签 (点击取消激活)",
                choices=[], # 初始为空，由 s_activated_tags_list 的值动态填充
                value=[],   # 值也由 s_activated_tags_list 控制
                interactive=True
            )

            gr.Markdown("### 筛选标签 (Filter Tags - Ranked by Search)")
            ranked_tags_checkbox_group = gr.CheckboxGroup(
                label="从搜索结果中选择标签进行激活",
                choices=[], # 初始为空，由 s_ranked_tags_from_api (排除已激活的) 动态填充
                value=[],   # 用于捕捉用户在此处勾选的标签，以便将其移至激活列表
                interactive=True
            )
            search_btn = gr.Button("搜索 (Search)", variant="primary")

        # --- 右侧栏: 搜索结果 ---
        with gr.Column(scale=3): # scale 控制列的相对宽度
            gr.Markdown("### 搜索结果 (Ranked Records)")
            results_output_markdown = gr.Markdown("尚未搜索，请配置参数后点击“搜索”按钮。")


    # --- UI 更新辅助函数 ---
    def format_records_to_markdown(records_list):
        """将API返回的 records 列表格式化为 Markdown 字符串"""
        if not records_list:
            return "没有找到相关结果。"

        output_md_parts = []
        for idx, record in enumerate(records_list):
            record_id = record.get('id', 'N/A')
            record_tags = record.get('tags', [])
            record_content = record.get('content', '无内容')

            # 将标签列表转换为 Markdown 的行内代码格式字符串
            tags_str = ", ".join([f"`{tag}`" for tag in record_tags]) if record_tags else "无标签"

            md_record = (
                f"**结果 {idx+1}**\n\n"
                f"- **ID:** {record_id}\n"
                f"- **标签 (Tags):** {tags_str}\n"
                f"- **内容 (Content):**\n```text\n{record_content}\n```\n"
                f"---\n"
            )
            output_md_parts.append(md_record)
        return "".join(output_md_parts)

    def generate_ui_updates_for_tag_checkbox_groups(current_activated_tags_py_list, current_api_ranked_tags_py_list):
        """
        根据当前的激活标签列表和API返回的ranked_tags列表，生成两个CheckboxGroup的更新对象。
        """
        # 更新 activated_tags_checkbox_group:
        # choices 和 value 都应该是当前实际激活的标签列表
        activated_cb_update = gr.update(choices=current_activated_tags_py_list, value=current_activated_tags_py_list)

        # 更新 ranked_tags_checkbox_group:
        # choices 应该是API返回的ranked_tags中，未被激活的那些标签
        visible_ranked_choices = [tag for tag in current_api_ranked_tags_py_list if tag not in current_activated_tags_py_list]
        # value 应重置为空列表，因为选择后即时处理
        ranked_cb_update = gr.update(choices=visible_ranked_choices, value=[])

        return activated_cb_update, ranked_cb_update

    # --- “搜索”按钮点击事件处理函数 ---
    def handle_search_button_click(query, mode, retrieval_k, rerank_strat, rerank_k, current_activated_tags_state_value):
        """
        处理搜索按钮点击事件。
        调用API，更新搜索结果显示，更新 s_ranked_tags_from_api 状态，并刷新两个标签选择框。
        """
        api_result = call_search_api(query, current_activated_tags_state_value, mode, retrieval_k, rerank_strat, rerank_k)

        if api_result.get("status") == "error":
            error_message = f"搜索出错: {api_result.get('message', '未知API错误')}"
            # 出错时，保持当前激活标签状态，清空API ranked tags状态，并更新显示
            act_cb_upd, rank_cb_upd = generate_ui_updates_for_tag_checkbox_groups(current_activated_tags_state_value, [])
            return (
                error_message,                      # results_output_markdown
                current_activated_tags_state_value, # s_activated_tags_list (state)
                [],                                 # s_ranked_tags_from_api (state)
                act_cb_upd,                         # activated_tags_checkbox_group (UI update)
                rank_cb_upd                         # ranked_tags_checkbox_group (UI update)
            )

        new_api_ranked_tags = api_result.get("ranked_tags", [])
        ranked_records = api_result.get("ranked_records", [])

        results_markdown_output = format_records_to_markdown(ranked_records)

        # s_activated_tags_list 状态本身在搜索后不直接改变，用户的激活选择会被保留。
        # s_ranked_tags_from_api 状态用新的API结果更新。
        # 然后根据这两个状态更新CheckboxGroup的显示。
        act_cb_upd, rank_cb_upd = generate_ui_updates_for_tag_checkbox_groups(current_activated_tags_state_value, new_api_ranked_tags)

        return (
            results_markdown_output,            # 更新Markdown结果区
            current_activated_tags_state_value, # s_activated_tags_list 状态值 (在本次搜索中不变)
            new_api_ranked_tags,                # 更新 s_ranked_tags_from_api 状态值
            act_cb_upd,                         # 更新 activated_tags_checkbox_group
            rank_cb_upd                         # 更新 ranked_tags_checkbox_group
        )
# pylint: disable=no-member # 忽略 Gradio 动态生成的属性警告
    search_btn.click(
        fn=handle_search_button_click,
        inputs=[
            query_input, mode_radio, retrieval_top_k_slider,
            rerank_strategy_dropdown, rerank_top_k_slider,
            s_activated_tags_list # 传入 s_activated_tags_list 的当前状态值
        ],
        outputs=[
            results_output_markdown,
            s_activated_tags_list,      # 输出到 s_activated_tags_list 状态 (尽管搜索本身可能不改变它)
            s_ranked_tags_from_api,     # 输出到 s_ranked_tags_from_api 状态
            activated_tags_checkbox_group, # 输出更新 activated_tags_checkbox_group 组件
            ranked_tags_checkbox_group  # 输出更新 ranked_tags_checkbox_group 组件
        ]
    )

    # --- 标签交互逻辑 ---

    # 当 ranked_tags_checkbox_group 中的标签被勾选 (激活操作)
    def handle_ranked_tag_selection_change(newly_selected_tags_in_ranked_group, current_activated_state_val, current_api_ranked_state_val):
        """
        处理 ranked_tags_checkbox_group 的值变化 (用户勾选了待激活标签)。
        将新勾选的标签加入 s_activated_tags_list 状态，并更新两个标签选择框。
        """
        # newly_selected_tags_in_ranked_group 是一个列表，包含在 ranked_tags_checkbox_group 中最新被勾选的标签

        # 合并当前已激活的标签和新选择要激活的标签，去重
        updated_activated_tags_list = sorted(list(set(current_activated_state_val + newly_selected_tags_in_ranked_group)))

        # 根据更新后的激活列表和API返回的ranked_tags列表，生成CheckboxGroup的更新
        act_cb_upd, rank_cb_upd = generate_ui_updates_for_tag_checkbox_groups(updated_activated_tags_list, current_api_ranked_state_val)

        return updated_activated_tags_list, act_cb_upd, rank_cb_upd

    ranked_tags_checkbox_group.change(
        fn=handle_ranked_tag_selection_change,
        inputs=[ranked_tags_checkbox_group, s_activated_tags_list, s_ranked_tags_from_api],
        outputs=[
            s_activated_tags_list,          # 更新 s_activated_tags_list 状态
            activated_tags_checkbox_group,  # 更新 activated_tags_checkbox_group 组件
            ranked_tags_checkbox_group      # 更新 ranked_tags_checkbox_group 组件 (choices改变，value清空)
        ]
    )

    # 当 activated_tags_checkbox_group 中的标签被取消勾选 (取消激活操作)
    def handle_activated_tag_deselection_change(current_selection_in_activated_group, current_api_ranked_state_val):
        """
        处理 activated_tags_checkbox_group 的值变化 (用户取消勾选了已激活标签)。
        直接使用 current_selection_in_activated_group 更新 s_activated_tags_list 状态，
        并据此刷新两个标签选择框。
        """
        # current_selection_in_activated_group 是 activated_tags_checkbox_group 中仍然保持勾选状态的标签列表
        # 这直接成为新的 s_activated_tags_list 状态值
        newly_set_activated_tags_list = current_selection_in_activated_group

        act_cb_upd, rank_cb_upd = generate_ui_updates_for_tag_checkbox_groups(newly_set_activated_tags_list, current_api_ranked_state_val)

        return newly_set_activated_tags_list, act_cb_upd, rank_cb_upd

    activated_tags_checkbox_group.change(
        fn=handle_activated_tag_deselection_change,
        inputs=[activated_tags_checkbox_group, s_ranked_tags_from_api],
        outputs=[
            s_activated_tags_list,          # 更新 s_activated_tags_list 状态
            activated_tags_checkbox_group,  # 更新 activated_tags_checkbox_group 组件
            ranked_tags_checkbox_group      # 更新 ranked_tags_checkbox_group 组件
        ]
    )

    # --- Demo 加载时的初始化操作 ---
    def initial_ui_load_setup(current_activated_state_val, current_api_ranked_state_val):
        """
        应用加载时调用，用于初始化标签选择框的显示。
        如果 s_ranked_tags_from_api 为空 (例如首次加载)，可以使用 initial_filter_tags_pool 作为初始 ranked tags。
        """
        # 如果 s_ranked_tags_from_api 状态为空 (比如是应用第一次加载，还没有任何搜索执行)
        # 并且我们希望 ranked_tags_checkbox_group 显示一些初始标签，则使用 initial_filter_tags_pool
        initial_ranked_tags_for_display = current_api_ranked_state_val
        if not current_api_ranked_state_val and initial_filter_tags_pool:
            initial_ranked_tags_for_display = list(initial_filter_tags_pool) # 使用副本

        # 根据（可能为空的）激活标签和（可能为初始默认或之前搜索结果的）ranked tags 更新UI
        act_cb_upd, rank_cb_upd = generate_ui_updates_for_tag_checkbox_groups(current_activated_state_val, initial_ranked_tags_for_display)

        # 这个函数也需要返回 s_ranked_tags_from_api 的新值，以防它被 initial_filter_tags_pool 初始化了
        return initial_ranked_tags_for_display, act_cb_upd, rank_cb_upd

    demo.load(
        fn=initial_ui_load_setup,
        inputs=[s_activated_tags_list, s_ranked_tags_from_api], # 传入当前状态值
        outputs=[s_ranked_tags_from_api, activated_tags_checkbox_group, ranked_tags_checkbox_group] # 更新状态和组件
    )

if __name__ == "__main__":
    # 确保 AdminCore.py 包含必要的 Milvus 连接设置或默认值
    VISITOR_WEBUI_HOST = os.getenv("VISITOR_WEBUI_HOST", "0.0.0.0")
    VISITOR_WEBUI_PORT = int(os.getenv("VISITOR_WEBUI_PORT", 7861))
    print(f"NaviSearch Admin Gradio UI 正在启动，访问地址: {VISITOR_WEBUI_HOST}:{VISITOR_WEBUI_PORT}")
    print(f"请确保您的 Admin FastAPI 服务正在运行在 {VISITOR_API_URL}")
    # 启动 Gradio 应用，指定服务器端口
    demo.launch(server_port=VISITOR_WEBUI_PORT)