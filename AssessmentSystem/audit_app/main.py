# audit_app/main.py
import gradio as gr # type: ignore # pylint: disable=import-error
import dotenv
import os
import sys
import functools # 用于 functools.partial
from typing import Optional
# 将项目根目录添加到 sys.path，以便可以正确解析 AssessmentSystem 的导入
# 这在使用 python -m audit_app.main 运行时通常不是必需的，
dotenv.load_dotenv()

# 但如果直接运行此脚本 (python audit_app/main.py)，则可能需要。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入应用模块
from AssessmentSystem.audit_app.models_ui import UIAssessmentReport
from AssessmentSystem.audit_app.app_utils import get_default_report_path, DEFAULT_INPUT_REPORT_DIR
from AssessmentSystem.audit_app import ui_components
from AssessmentSystem.audit_app import callbacks

# 尝试导入核心客户端，并处理可能的缺失
try:
    from AssessmentSystem.llm_client import LLMAssessmentClient # pylint: disable=import-error
    from AssessmentSystem.navi_search_client import NaviSearchClient # pylint: disable=import-error
    clients_available = True
except ImportError:
    print("警告: main.py - LLMClient 或 NaviSearchClient 无法从 AssessmentSystem 导入。")
    print("请确保 AssessmentSystem 包已正确安装或在 PYTHONPATH 中。")
    print("搜索和LLM功能将不可用。")
    LLMAssessmentClient = None # type: ignore
    NaviSearchClient = None # type: ignore
    clients_available = False

# --- 全局客户端实例化 ---
# 这些客户端将在 Gradio 应用启动时创建一次。
llm_client_instance: Optional[LLMAssessmentClient] = None
navi_search_client_instance: Optional[NaviSearchClient] = None

def initialize_clients():
    """初始化 LLM 和 NaviSearch 客户端。"""
    global llm_client_instance, navi_search_client_instance # pylint: disable=global-statement
    if not clients_available:
        print("核心客户端模块不可用，跳过客户端初始化。")
        return

    try:
        # 从环境变量加载配置 (llm_client 和 navi_search_client 应该支持这个)
        # 例如: LLM_BASE_URL, LLM_API_KEY, ADMIN_API_HOST, VISITOR_API_PORT 等
        # 如果 .env 文件存在并且被 python-dotenv 加载 (通常在客户端内部完成)，这里应该能工作。
        print("正在初始化 LLMAssessmentClient...")
        llm_client_instance = LLMAssessmentClient() # 假设构造函数可以处理 .env
        print("LLMAssessmentClient 初始化成功。")

        print("正在初始化 NaviSearchClient...")
        # NaviSearchClient 需要 admin_url, visitor_url, collection_name, 和 llm_client
        # 这些应该从环境变量或配置文件中获取
        admin_host = os.getenv("ADMIN_API_HOST", "localhost")
        admin_port = os.getenv("ADMIN_API_PORT", "8001") # 假设默认端口
        visitor_host = os.getenv("VISITOR_API_HOST", "localhost")
        visitor_port = os.getenv("VISITOR_API_PORT", "8000") # 假设默认端口
        collection_name = os.getenv("NAVI_COLLECTION_NAME", "my_evidence_collection")

        admin_api_url = f"http://{admin_host}:{admin_port}"
        visitor_api_url = f"http://{visitor_host}:{visitor_port}"

        if not llm_client_instance: # 如果LLM客户端未能初始化
            raise ValueError("LLMClient 必须在 NaviSearchClient 之前初始化。")
        print(admin_api_url)
        print(visitor_api_url)
        navi_search_client_instance = NaviSearchClient(
            admin_url=admin_api_url,
            visitor_url=visitor_api_url,
            evidence_collection_name=collection_name,
            llm_client=llm_client_instance,
            insert_evidences = True
        )
        print("NaviSearchClient 初始化成功。")

    except ValueError as ve: # 通常是由于缺少环境变量导致的配置错误
        print(f"客户端初始化错误 (ValueError): {ve}")
        print("请检查您的 .env 文件或环境变量是否已正确设置 LLM_BASE_URL, LLM_MODEL, API URLs 等。")
        llm_client_instance = None
        navi_search_client_instance = None
    except ImportError: # 如果 dotenv 等未安装，或者内部导入失败
        print("客户端初始化时发生导入错误。请确保所有依赖项已安装。")
        llm_client_instance = None
        navi_search_client_instance = None
    except Exception as e: # pylint: disable=broad-except
        print(f"初始化客户端时发生未知错误: {e}")
        llm_client_instance = None
        navi_search_client_instance = None

    # 将客户端实例注入到回调函数 (如果它们被定义为需要这些属性)
    if hasattr(callbacks, 'handle_regenerate_conclusion_click'):
        setattr(callbacks.handle_regenerate_conclusion_click, 'llm_client', llm_client_instance)
    if hasattr(callbacks, 'handle_search_evidence_click'):
        setattr(callbacks.handle_search_evidence_click, 'navi_search_client', navi_search_client_instance)


# --- Gradio 应用构建 ---
def build_audit_app_ui():
    """构建并返回Gradio Blocks界面。"""
    with gr.Blocks(theme=gr.themes.Soft(), title="网络安全评估报告审计工具") as app: # type: ignore # pylint: disable=no-member
        # 1. 应用状态 (存储整个 UIAssessmentReport 对象)
        # 初始值为 None，在加载报告后填充
        report_state = gr.State(value=None) # type: ignore # pylint: disable=no-member

        # 2. 顶部区域: 文件加载
        with gr.Row(): # type: ignore # pylint: disable=no-member
            report_file_input = gr.File(label="选择评估报告JSON文件进行加载", file_count="single", file_types=[".json"], type="filepath") # type: ignore # pylint: disable=no-member
            # 或者使用 Textbox 输入路径
            # report_path_tb = gr.Textbox(label="或输入报告文件路径", placeholder=get_default_report_path())
            load_report_btn = gr.Button("📂 加载报告", variant="primary") # type: ignore # pylint: disable=no-member

        # 3. 主内容区: 左右分栏
        # 使用 Slider 控制左右栏宽度比例
        # Pylint 可能会抱怨 gr.Column 的 scale 参数，因为它是动态的
        # mypy 可能会抱怨 gr.Column 不是一个有效的上下文管理器，如果 types-gradio 不完整

        # 初始比例，例如左60%，右40%
        initial_left_scale = 6
        initial_right_scale = 4

        gr.Markdown("---") # type: ignore # pylint: disable=no-member

        # 定义列容器
        # 我们将把列的创建放在一个函数中，以便在需要时可以重新创建它们（虽然不理想）
        # 或者，更简单地，我们接受Gradio的默认布局行为，它会尝试平均分配空间。
        # 对于分栏，gr.Row > gr.Column 是标准做法。

        with gr.Row(equal_height=False): # type: ignore # pylint: disable=no-member
            with gr.Column(scale=initial_left_scale, min_width=400) as left_panel_col: # type: ignore # pylint: disable=no-member
                report_detail_components = ui_components.create_report_detail_module()

            with gr.Column(scale=initial_right_scale, min_width=350) as right_panel_col: # type: ignore # pylint: disable=no-member
                search_module_components = ui_components.create_evidence_search_module()

        gr.Markdown("---") # type: ignore # pylint: disable=no-member
        # 4. 主导航栏
        nav_components = ui_components.create_top_nav_bar()
        # 5. 统计信息显示
        statistics_components = ui_components.create_statistics_display()

        # 6. 底部状态栏
        status_bar = ui_components.create_status_bar()

        # 7. 收集所有输出组件到一个列表，以便回调函数可以更新它们
        # 顺序必须与回调函数返回的 gr.update() 元组的顺序严格一致。
        all_ui_outputs = [
            report_state, # 第一个总是主状态
            status_bar,
            nav_components["progress_display"],
            # 左侧报告详情模块的组件 (按 update_report_detail_module_components 返回顺序)
            report_detail_components["spec_id"],
            report_detail_components["spec_content"],
            report_detail_components["judgement"],
            report_detail_components["comment"],
            report_detail_components["referenced_evidences_display"],
            report_detail_components["referenced_evidence_selector"],
            report_detail_components["ref_evidence_actions_row"],
            report_detail_components["llm_comparison_area"], # Column
            report_detail_components["llm_compare_current_judgement"],
            report_detail_components["llm_compare_current_comment"],
            report_detail_components["llm_compare_suggested_judgement"],
            report_detail_components["llm_compare_suggested_comment"],
            # 右侧证据搜索模块的组件 (按 update_evidence_search_module_components 返回顺序, 第一个是查询框，通常作为输入)
            search_module_components["search_query_tb"], # 通常是输入，但也可能被回调更新（例如，加载项时）
            search_module_components["selected_filter_tags_container"],
            search_module_components["recommended_tags_container"],
            search_module_components["search_results_display"],
            search_module_components["search_result_selector"],
            search_module_components["search_result_actions_row"],
            # 统计组件
            statistics_components["judgement_stats_md"],
            statistics_components["audit_status_stats_md"],
            # LLM对比区的按钮的激活状态等也可以作为输出，如果需要动态控制
            report_detail_components["adopt_llm_btn"], # 可能需要更新其 .interactive 状态
            report_detail_components["discard_llm_btn"]
        ]

        # --- 绑定回调函数 ---

        # 加载报告按钮
        # 输出: report_state, status_bar, progress_display, 左侧组件..., 右侧组件..., 统计组件...
        # 确保输出列表与 handle_load_report_click 返回的元组匹配
        load_report_outputs = [
            report_state,
            status_bar,
            nav_components["progress_display"],
            # *report_detail_components.values(), # 按字典顺序解包，这可能不安全，最好显式列出
            # 显式列出左侧组件的输出顺序
            report_detail_components["spec_id"],
            report_detail_components["spec_content"],
            report_detail_components["judgement"],
            report_detail_components["comment"],
            report_detail_components["referenced_evidences_display"],
            report_detail_components["referenced_evidence_selector"],
            report_detail_components["ref_evidence_actions_row"],
            report_detail_components["llm_comparison_area"],
            report_detail_components["llm_compare_current_judgement"],
            report_detail_components["llm_compare_current_comment"],
            report_detail_components["llm_compare_suggested_judgement"],
            report_detail_components["llm_compare_suggested_comment"],
            # 显式列出右侧组件的输出顺序 (与 update_evidence_search_module_components 对应)
            search_module_components["search_query_tb"], # 第一个输出是查询框本身
            search_module_components["selected_filter_tags_container"],
            search_module_components["recommended_tags_container"],
            search_module_components["search_results_display"],
            search_module_components["search_result_selector"],
            search_module_components["search_result_actions_row"],
            # 统计组件
            statistics_components["judgement_stats_md"],
            statistics_components["audit_status_stats_md"]
        ]

        load_report_btn.click(
            fn=callbacks.handle_load_report_click,
            inputs=[report_file_input], # report_path_tb
            outputs=load_report_outputs
        )

        # 导航按钮的共同输入和输出结构
        nav_inputs = [
            report_state,
            report_detail_components["judgement"], # 当前判断结论 (str)
            report_detail_components["comment"]    # 当前评论 (str)
        ]
        # 输出与 load_report_outputs 结构相同 (除了第一个输入 report_file_input)
        nav_outputs = load_report_outputs
        print("nav_outputs")
        nav_components["prev_btn"].click(
            fn=callbacks.handle_prev_item_click,
            inputs=nav_inputs,
            outputs=nav_outputs
        )
        nav_components["skip_btn"].click(
            fn=callbacks.handle_skip_item_click,
            inputs=[report_state], # Skip 不需要保存当前结论
            outputs=nav_outputs
        )
        nav_components["next_btn"].click(
            fn=callbacks.handle_next_item_click,
            inputs=nav_inputs,
            outputs=nav_outputs
        )
        nav_components["save_btn"].click(
            fn=callbacks.handle_save_work_click,
            inputs=nav_inputs, # 保存时也需要收集当前UI上的结论
            outputs=[report_state, status_bar] # 只更新状态和状态栏
        )

        # 引用证据操作按钮
        report_detail_components["toggle_evidence_btn"].click(
            fn=callbacks.handle_toggle_reference_evidence,
            inputs=[report_state, report_detail_components["referenced_evidence_selector"]],
            outputs=[
                report_state, status_bar,
                report_detail_components["referenced_evidences_display"]
            ]
        )
        report_detail_components["remove_evidence_btn"].click(
            fn=callbacks.handle_remove_reference_evidence,
            inputs=[report_state, report_detail_components["referenced_evidence_selector"]],
            outputs=[
                report_state, status_bar,
                report_detail_components["referenced_evidences_display"],
                report_detail_components["referenced_evidence_selector"],
                report_detail_components["ref_evidence_actions_row"]
            ]
        )

        # 证据搜索模块的回调
        # 添加/移除过滤标签
        # 输出: status_bar, selected_filter_tags_container, recommended_tags_container, search_results_display, search_result_selector, search_result_actions_row
        search_filter_tags_action_outputs = [
            status_bar,
            search_module_components["selected_filter_tags_container"],
            search_module_components["recommended_tags_container"], # 保持不变，但仍需作为输出占位
            search_module_components["search_results_display"],       # 保持不变
            search_module_components["search_result_selector"],     # 保持不变
            search_module_components["search_result_actions_row"]   # 保持不变
        ]
        search_module_components["add_filter_tag_btn"].click(
            fn=callbacks.handle_add_filter_tag,
            inputs=[report_state, search_module_components["filter_tag_input_tb"], search_module_components["search_query_tb"]],
            outputs=search_filter_tags_action_outputs
        )
        search_module_components["remove_filter_tag_btn"].click(
            fn=callbacks.handle_remove_filter_tag,
            inputs=[report_state, search_module_components["filter_tag_input_tb"], search_module_components["search_query_tb"]],
            outputs=search_filter_tags_action_outputs
        )
        # 执行搜索按钮
        search_module_components["search_btn"].click(
            fn=callbacks.handle_search_evidence_click,
            inputs=[report_state, search_module_components["search_query_tb"]],
            outputs=search_filter_tags_action_outputs # 同样的输出结构，但内容会更新
        )
        # 引用搜索到的证据按钮
        search_module_components["reference_searched_evidence_btn"].click(
            fn=callbacks.handle_reference_searched_evidence,
            inputs=[report_state, search_module_components["search_result_selector"]],
            outputs=[ # 更新 report_state, status_bar, 和左侧的引用证据区
                report_state, status_bar,
                report_detail_components["referenced_evidences_display"],
                report_detail_components["referenced_evidence_selector"],
                report_detail_components["ref_evidence_actions_row"]
            ]
        )

        # LLM 相关按钮
        nav_components["regenerate_btn"].click(
            fn=callbacks.handle_regenerate_conclusion_click,
            inputs=[report_state],
            outputs=[ # report_state, status_bar, LLM对比区及其内部组件
                report_state, status_bar,
                report_detail_components["llm_comparison_area"],
                report_detail_components["llm_compare_current_judgement"],
                report_detail_components["llm_compare_current_comment"],
                report_detail_components["llm_compare_suggested_judgement"],
                report_detail_components["llm_compare_suggested_comment"]
            ]
        )
        report_detail_components["adopt_llm_btn"].click(
            fn=callbacks.handle_adopt_llm_suggestion,
            inputs=[report_state],
            outputs=[ # report_state, status_bar, judgement, comment, ref_ev_display, llm_area_visibility, stat_judgement, stat_audit
                report_state, status_bar,
                report_detail_components["judgement"],
                report_detail_components["comment"],
                report_detail_components["referenced_evidences_display"],
                report_detail_components["llm_comparison_area"], # 更新可见性
                statistics_components["judgement_stats_md"],
                statistics_components["audit_status_stats_md"]
            ]
        )
        report_detail_components["discard_llm_btn"].click(
            fn=callbacks.handle_discard_llm_suggestion,
            inputs=[report_state],
            outputs=[report_state, status_bar, report_detail_components["llm_comparison_area"]]
        )

        # 当应用加载时，尝试自动加载默认报告 (如果存在)
        # app.load() 仅在 Blocks.launch() 之前调用，用于设置初始状态。
        # 这里我们用一个隐藏按钮或 app.load 事件来触发初始加载。
        # Gradio 的 app.load() 似乎更适合整个页面加载时的事件。
        # 为了简化，我们让用户手动点击“加载报告”按钮。
        # 如果需要自动加载，可以考虑：
        # def auto_load_on_start():
        #     return callbacks.handle_load_report_click(get_default_report_path())
        # app.load(auto_load_on_start, inputs=None, outputs=load_report_outputs)
        # 但要注意 app.load 的输入输出与回调的匹配。

    return app

if __name__ == "__main__":
    print("正在启动网络安全评估报告审计工具...")
    initialize_clients() # 初始化 LLM 和 NaviSearch 客户端

    if not clients_available:
        print("\n警告：由于核心客户端未能加载，部分功能（如证据搜索、LLM结论生成）将无法使用。")
        print("应用仍会启动，但功能受限。请检查依赖和配置。\n")
    elif not llm_client_instance or not navi_search_client_instance:
        print("\n警告：一个或多个后端客户端未能成功初始化。")
        print("证据搜索和/或LLM结论生成功能可能无法正常工作。\n")


    audit_app_interface = build_audit_app_ui()

    # 启动 Gradio 应用
    # share=True 会创建一个公开链接 (如果通过 Hugging Face Spaces 或类似服务部署)
    # debug=True 会在浏览器控制台显示更多调试信息
    audit_app_interface.launch(share=False, debug=True, server_name="0.0.0.0", server_port=7862)
    # 使用 server_name="0.0.0.0" 使其可以从本地网络访问
    # server_port 可以指定端口，默认为 7860
