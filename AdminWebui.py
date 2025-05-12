# NaviSearchAdminWebui.py
# Milvus Collection 和 Document 管理的 Gradio Web UI

import sys
import gradio as gr
import requests
import json
import traceback # 用于更详细的错误日志输出
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncIterator
# FastAPI 后端服务地址
# 请确保这里的地址和端口与您的 AdminFastAPI.py 中 uvicorn.run 配置的一致
ADMIN_FASTAPI_URL = "http://localhost:8001"

# Milvus 向量嵌入维度
# 需要与后端 AdminCore 中的 DEFAULT_DIM 保持一致，用于前端输入提示和验证
DEFAULT_DIM = 1024 # TODO: 考虑从后端 API 获取此值以确保一致性

# --- API 调用函数 ---

def call_admin_api(endpoint: str, method: str = "GET", json_payload: dict = None):
    """
    通用的 Admin API 调用函数。
    Args:
        endpoint: API 路径，例如 "/collections" 或 "/collections/init"
        method: HTTP 方法，例如 "GET", "POST", "DELETE"
        json_payload: 请求体，字典格式，用于 POST 方法

    Returns:
        API 返回的 JSON 响应字典，或包含错误信息的字典。
    """
    url = f"{ADMIN_FASTAPI_URL}{endpoint}"
    try:
        print(f"调用 API: {method} {url}")
        if json_payload is not None:
             print(f"请求体: {json.dumps(json_payload, indent=2, ensure_ascii=False)}")

        if method == "GET":
            # 对于 GET 请求，json 参数通常会被添加到 URL query string 中，
            # 但考虑到 FastAPI 可能支持 GET 请求体，这里保留 json 参数
            response = requests.get(url, json=json_payload)
        elif method == "POST":
            response = requests.post(url, json=json_payload)
        elif method == "DELETE":
            # DELETE 方法通常通过 URL 路径或 query string 传递参数，请求体通常为空
            response = requests.delete(url)
        else:
            return {"status": "error", "message": f"不支持的 HTTP 方法: {method}"}

        # 如果 HTTP 请求返回了错误状态码 (4xx 或 5xx)，则抛出 HTTPError 异常
        response.raise_for_status()

        # 尝试解析 JSON 响应
        try:
            return response.json()
        except json.JSONDecodeError:
            # 如果状态码是 204 No Content (例如成功删除操作)，response.json() 会抛出错误
            # 此时认为操作成功但没有返回内容
            if response.status_code == 204:
                return {"status": "success", "message": "操作成功，无返回内容。"}
            # 如果响应不是 JSON，返回错误信息和原始响应文本
            return {"status": "error", "message": f"API 返回非 JSON 响应 (状态码: {response.status_code})。响应文本: {response.text}"}

    except requests.exceptions.ConnectionError as e:
        # 处理连接错误 (例如后端服务未运行)
        error_msg = f"API 请求失败: 无法连接到 {url}。请检查后端服务是否正在运行。错误详情: {e}"
        print(f"连接错误: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": error_msg}
    except requests.exceptions.HTTPError as e:
        # 处理 HTTP 错误 (例如 404, 500)
        status_code = e.response.status_code
        reason = e.response.reason
        response_text = e.response.text
        error_msg = f"API 请求失败: HTTP 错误 - {status_code} {reason}。"
        print(f"HTTP 错误: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # 尝试从 HTTP 错误响应中提取更详细的错误信息 (如果后端 FastAPI 提供了 detail 字段)
        try:
            error_detail = e.response.json().get('detail', response_text)
            error_msg = f"API 请求失败: HTTP 错误 - {status_code} {reason}。详情: {error_detail}"
        except json.JSONDecodeError:
            pass # 如果响应不是 JSON，则使用默认错误信息
        return {"status": "error", "message": error_msg}
    except requests.exceptions.Timeout:
        # 处理请求超时
        error_msg = f"API 请求超时: 请求 {url} 超时。"
        print(f"超时错误: {error_msg}", file=sys.stderr)
        return {"status": "error", "message": error_msg}
    except requests.exceptions.RequestException as e:
        # 处理其他 requests 库相关的错误
        error_msg = f"API 请求过程中发生未知错误: {e}"
        print(f"请求异常: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        # 处理调用函数时发生的其他意外错误
        error_msg = f"调用 API 函数时发生意外错误: {str(e)}"
        print(f"意外错误: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": error_msg}


# --- Gradio 应用定义 ---
# pylint: disable=no-member # 忽略 Gradio 动态生成的属性警告
# 使用 gr.Blocks 创建一个更灵活的布局
with gr.Blocks(title="NaviSearch Admin 管理界面", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## NaviSearch Admin 管理界面")
    gr.Markdown(f"连接到后端 FastAPI 服务: {ADMIN_FASTAPI_URL}")

    # 状态变量用于存储当前 Collection 列表 (虽然界面上显示，但后端状态更准确)
    # s_collection_list = gr.State([]) # 暂不需要在 Gradio 状态中维护

    with gr.Row():
        # --- 左侧栏: 操作区域 ---
        with gr.Column(scale=1):
            gr.Markdown("### 集合管理 (Collection Management)")
            # 单独的 Collection 名称输入框，供多个操作使用
            collection_name_input = gr.Textbox(label="Collection 名称", placeholder="输入 Collection 名称...", info="例如: my_documents")

            with gr.Accordion("基础操作", open=True): # 默认展开
                create_collection_btn = gr.Button("创建 Collection")
                drop_collection_btn = gr.Button("删除 Collection")
                list_collections_btn = gr.Button("列出所有 Collections")
                has_collection_btn = gr.Button("检查 Collection 是否存在")

            with gr.Accordion("高级操作", open=False): # 默认折叠
                init_collection_drop_checkbox = gr.Checkbox(label="初始化前先删除现有 Collection", value=False, info="勾选此项将删除同名 Collection 后再创建")
                # 自定义 Schema 输入
                custom_schema_input = gr.Textbox(
                    label="自定义 Collection Schema (JSON 格式)",
                    placeholder='留空则使用默认 Schema。示例: {"fields": [{"name": "id", "dtype": "INT64", "is_primary": true, "auto_id": true}, {"name": "title", "dtype": "VARCHAR", "max_length": 256}, {"name": "content", "dtype": "VARCHAR", "max_length": 65535}, {"name": "tags", "dtype": "JSON"}, {"name": "embedding", "dtype": "FLOAT_VECTOR", "dim": 1024}], "description": "我的自定义 Schema"}',
                    lines=5,
                    info="定义 Collection 的字段结构。请参考 Milvus FieldSchema 和 CollectionSchema 定义。"
                )
                # 索引参数输入
                index_field_name_input = gr.Textbox(label="要创建索引的向量字段名称", value="embedding", placeholder="通常是 embedding", info="指定在哪个向量字段上创建索引")
                index_params_input = gr.Textbox(
                    label="索引参数 (JSON 格式)",
                    placeholder='留空则使用默认索引参数 (AUTOINDEX, COSINE)。示例: {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}',
                    lines=3,
                    info="定义向量索引的类型和参数。请参考 Milvus 索引文档。"
                )
                init_collection_btn = gr.Button("初始化 Collection 结构 (创建并建索引)")

                create_index_btn = gr.Button("为 Collection 创建索引")
                load_collection_btn = gr.Button("加载 Collection 到内存")


            gr.Markdown("### 记录插入 (Record Insertion)")
            # 单独的插入目标 Collection 名称输入框
            insert_collection_name_input = gr.Textbox(label="目标 Collection 名称", placeholder="输入要插入记录的 Collection 名称...")

            with gr.Accordion("插入单条记录", open=True):
                record_content_input = gr.Textbox(label="内容 (content)", placeholder="记录的文本内容...")
                record_tags_input = gr.Textbox(label="标签 (tags, JSON 格式)", placeholder='例如: ["标签A", "标签B"]', value="[]", info="必须是有效的 JSON 数组。")
                # Embedding 输入框，如果留空且后端支持，将自动从 content 生成
                record_embedding_input = gr.Textbox(label=f"向量嵌入 (embedding, JSON 浮点数列表, 维度 {DEFAULT_DIM} 可选)", placeholder=f'例如: [{0.1}, {0.2}, ...]。留空则后端尝试从内容自动生成。', lines=3, info="必须是有效的 JSON 浮点数数组，维度需与 Collection Schema 匹配。")
                # 后端 AdminCore 负责决定是否生成 embedding，前端不提供 auto_generate_embedding 勾选框，
                # 留空 embedding 字段即表示希望自动生成（如果后端配置允许）。
                insert_one_btn = gr.Button("插入单条记录")

            with gr.Accordion("批量插入记录", open=False):
                # 批量插入可以通过文件上传或文本区域输入 JSON 数组
                # 期望的 JSON 数组格式示例: [{"content": "...", "tags": [...]}, {"content": "...", "tags": [...], "embedding": [...]}, ...]
                bulk_insert_file = gr.File(label="上传 JSON 文件批量插入", file_types=[".json"])
                bulk_insert_json_text = gr.Textbox(label="或直接输入 JSON 数组进行批量插入", placeholder='例如: [{"content": "...", "tags": [...]}, ...]', lines=5, info="输入符合格式要求的 JSON 记录数组。")
                # 批量插入的 auto_generate_embedding 参数由后端控制，前端不暴露
                # 批量插入的 batch_size 参数由后端 AdminCore 控制，前端不暴露
                insert_many_btn = gr.Button("批量插入记录")


        # --- 右侧栏: 结果显示区域 ---
        with gr.Column(scale=2):
            gr.Markdown("### 操作结果 (Operation Results)")
            # 显示 API 调用的原始 JSON 响应或错误信息
            results_output = gr.Textbox(label="API 响应", interactive=False, lines=25, max_lines=25)
            # 单独显示 Collection 列表，方便查看
            collection_list_output = gr.Textbox(label="Collections 列表", interactive=False, lines=5, max_lines=5, show_copy_button=True)


    # --- 事件处理函数 ---

    # Helper function to format and display API result in the output textbox
    def display_api_result(result: Any) -> str:
        """将 API 响应结果格式化为 JSON 字符串显示。"""
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, indent=2, ensure_ascii=False)
            except TypeError:
                 # Fallback if object is not JSON serializable
                 return str(result)
        return str(result)

    # Collection Management Handlers

    def handle_create_collection(collection_name: str, custom_schema_str: str):
        """处理创建 Collection 操作。"""
        if not collection_name:
            return "错误: Collection 名称不能为空。", gr.update(visible=True) # 返回错误信息

        json_payload = {"collection_name": collection_name}
        if custom_schema_str:
            try:
                custom_schema_data = json.loads(custom_schema_str)
                # 在发送给 AdminCore 前，前端无需将 JSON 转为 Pymilvus 对象，AdminCore 会处理
                json_payload["custom_schema"] = custom_schema_data
            except json.JSONDecodeError:
                return "错误: 自定义 Schema JSON 格式无效。", gr.update(visible=True)
            except Exception as e:
                 return f"处理自定义 Schema 时发生错误: {str(e)}", gr.update(visible=True)

        result = call_admin_api("/collections/create", method="POST", json_payload=json_payload)
        return display_api_result(result), gr.update(visible=True)


    def handle_drop_collection(collection_name: str):
        """处理删除 Collection 操作。"""
        if not collection_name:
            return "错误: Collection 名称不能为空。", gr.update(visible=True)
        # DELETE 方法通过 URL 路径传递名称
        result = call_admin_api(f"/collections/{collection_name}", method="DELETE")
        return display_api_result(result), gr.update(visible=True)


    def handle_list_collections():
        """处理列出所有 Collections 操作。"""
        result = call_admin_api("/collections", method="GET")
        result_text = display_api_result(result)
        if result.get("status") == "success":
            collection_names = result.get("collections", [])
            # 返回给 Gradio 两个输出框的内容
            return result_text, ", ".join(collection_names) if collection_names else "无 Collections", gr.update(visible=True)
        else:
            # 如果 API 调用失败，Collection 列表显示错误信息
            return result_text, result.get("message", "获取列表失败"), gr.update(visible=True)


    def handle_has_collection(collection_name: str):
        """处理检查 Collection 是否存在操作。"""
        if not collection_name:
            return "错误: Collection 名称不能为空。", gr.update(visible=True)
        result = call_admin_api(f"/collections/{collection_name}/exists", method="GET")
        return display_api_result(result), gr.update(visible=True)


    def handle_init_collection(
        collection_name: str,
        drop_existing: bool,
        custom_schema_str: str,
        index_field_name: str,
        index_params_str: str
    ):
        """处理初始化 Collection 结构操作。"""
        if not collection_name:
            return "错误: Collection 名称不能为空。", gr.update(visible=True)
        if not index_field_name:
             return "错误: 索引字段名称不能为空。", gr.update(visible=True)

        json_payload: Dict[str, Any] = {
            "collection_name": collection_name,
            "drop_existing": drop_existing,
            "vector_field_name": index_field_name
        }

        # 处理自定义 schema
        if custom_schema_str:
            try:
                custom_schema_data = json.loads(custom_schema_str)
                # 验证基本结构，确保 fields 是列表
                if not isinstance(custom_schema_data, dict) or not isinstance(custom_schema_data.get("fields"), list):
                     return "错误: 自定义 Schema JSON 格式无效或缺少 'fields' 列表。", gr.update(visible=True)
                json_payload["custom_schema"] = custom_schema_data
            except json.JSONDecodeError:
                return "错误: 自定义 Schema JSON 格式无效。", gr.update(visible=True)
            except Exception as e:
                 return f"处理自定义 Schema 时发生错误: {str(e)}", gr.update(visible=True)


        # 处理索引参数
        if index_params_str:
            try:
                index_params_data = json.loads(index_params_str)
                if not isinstance(index_params_data, dict):
                     return "错误: 索引参数必须是有效的 JSON 对象 (字典)。", gr.update(visible=True)
                json_payload["index_params"] = index_params_data
            except json.JSONDecodeError:
                return "错误: 索引参数 JSON 格式无效。", gr.update(visible=True)
            except Exception as e:
                 return f"处理索引参数时发生错误: {str(e)}", gr.update(visible=True)

        result = call_admin_api("/collections/init", method="POST", json_payload=json_payload)
        return display_api_result(result), gr.update(visible=True)


    def handle_create_index(collection_name: str, field_name: str, index_params_str: str):
        """处理为 Collection 创建索引操作。"""
        if not collection_name:
            return "错误: Collection 名称不能为空。", gr.update(visible=True)
        if not field_name:
            return "错误: 索引字段名称不能为空。", gr.update(visible=True)

        json_payload: Dict[str, Any] = {
            "collection_name": collection_name,
            "field_name": field_name,
        }
        if index_params_str:
            try:
                index_params_data = json.loads(index_params_str)
                if not isinstance(index_params_data, dict):
                     return "错误: 索引参数必须是有效的 JSON 对象 (字典)。", gr.update(visible=True)
                json_payload["index_params"] = index_params_data
            except json.JSONDecodeError:
                return "错误: 索引参数 JSON 格式无效。", gr.update(visible=True)
            except Exception as e:
                 return f"处理索引参数时发生错误: {str(e)}", gr.update(visible=True)


        result = call_admin_api("/collections/index", method="POST", json_payload=json_payload)
        return display_api_result(result), gr.update(visible=True)


    def handle_load_collection(collection_name: str):
        """处理加载 Collection 到内存操作。"""
        if not collection_name:
            return "错误: Collection 名称不能为空。", gr.update(visible=True)
        json_payload = {"collection_name": collection_name}
        result = call_admin_api("/collections/load", method="POST", json_payload=json_payload)
        return display_api_result(result), gr.update(visible=True)


    # Record Insertion Handlers

    def handle_insert_one(
        collection_name: str,
        content: str,
        tags_str: str,
        embedding_str: str
    ):
        """处理插入单条记录操作。"""
        if not collection_name:
            return "错误: 目标 Collection 名称不能为空。", gr.update(visible=True)

        # 验证并解析 tags
        tags_data = []
        if tags_str:
            try:
                tags_data = json.loads(tags_str)
                if not isinstance(tags_data, list):
                     return "错误: 标签 (tags) 必须是有效的 JSON 数组。", gr.update(visible=True)
            except json.JSONDecodeError:
                return "错误: 标签 (tags) JSON 格式无效。", gr.update(visible=True)
            except Exception as e:
                 return f"处理标签时发生错误: {str(e)}", gr.update(visible=True)

        # 验证并解析 embedding
        embedding_data = None
        if embedding_str:
            try:
                embedding_data = json.loads(embedding_str)
                # 仅验证格式，维度一致性由后端 AdminCore 检查
                if not isinstance(embedding_data, list) or not all(isinstance(x, (int, float)) for x in embedding_data):
                     return "错误: 向量嵌入 (embedding) 必须是有效的 JSON 浮点数列表。", gr.update(visible=True)
                # 前端可以提示用户维度，但不强制验证，依赖后端
                # if embedding_data and len(embedding_data) != DEFAULT_DIM:
                #     return f"警告: 向量嵌入维度 ({len(embedding_data)}) 与默认维度 ({DEFAULT_DIM}) 不一致。请确保与 Collection Schema 匹配。", gr.update(visible=True)

            except json.JSONDecodeError:
                return "错误: 向量嵌入 (embedding) JSON 格式无效。", gr.update(visible=True)
            except Exception as e:
                 return f"处理向量嵌入时发生错误: {str(e)}", gr.update(visible=True)


        # 构建 record_data 字典
        record_data: Dict[str, Any] = {
            "content": content,
            "tags": tags_data,
        }
        # 仅当用户提供了合法的 embedding 字符串时，才将其添加到 payload 中
        # 后端会根据 payload 中是否存在 embedding 字段来决定是否自动生成
        if embedding_str and embedding_data is not None: # 确保 embedding_data 已成功解析
             record_data["embedding"] = embedding_data

        # 如果 content 为空且用户未提供 embedding，给予提示
        if not content and "embedding" not in record_data:
             return "警告: 内容 (content) 和向量嵌入 (embedding) 都未提供。如果后端未配置自动生成或 Collection 需要 embedding，插入可能失败。", gr.update(visible=True)


        json_payload = {"collection_name": collection_name, "record_data": record_data}

        result = call_admin_api("/records/insert_one", method="POST", json_payload=json_payload)
        return display_api_result(result), gr.update(visible=True)


    def handle_insert_many(
        collection_name: str,
        file_obj: Optional[gr.File],
        json_text: str
    ):
        """处理批量插入记录操作。"""
        if not collection_name:
            return "错误: 目标 Collection 名称不能为空。", gr.update(visible=True)

        records_to_insert: List[Dict[str, Any]] = []
        error_message = None

        if file_obj is not None:
            # 处理文件上传
            try:
                # 读取文件内容并解析 JSON
                file_content = file_obj.read().decode('utf-8')
                records_to_insert = json.loads(file_content)
                if not isinstance(records_to_insert, list):
                    error_message = "错误: 上传的文件内容必须是有效的 JSON 数组。"
            except json.JSONDecodeError:
                error_message = "错误: 上传的文件不是有效的 JSON 格式。"
            except Exception as e:
                 error_message = f"读取上传文件时发生错误: {str(e)}"
        elif json_text:
            # 处理文本框输入的 JSON
            try:
                records_to_insert = json.loads(json_text)
                if not isinstance(records_to_insert, list):
                    error_message = "错误: 文本框输入的内容必须是有效的 JSON 数组。"
            except json.JSONDecodeError:
                error_message = "错误: 文本框输入的 JSON 格式无效。"
            except Exception as e:
                 error_message = f"处理文本输入时发生错误: {str(e)}"
        else:
            error_message = "错误: 请上传一个 JSON 文件或在文本框中输入 JSON 数组进行批量插入。"

        # 如果有前置错误，直接返回
        if error_message:
             return error_message, gr.update(visible=True)

        if not records_to_insert:
            return "提示: 没有要插入的记录。", gr.update(visible=True)

        # 对每条记录进行基本验证 (content 和 tags 存在，tags 是列表，可选的 embedding 格式)
        # 这里的验证与后端 AdminCore 的要求一致，但后端会更严格
        for i, record in enumerate(records_to_insert):
            if not isinstance(record, dict):
                 return f"错误: 批量插入数据中索引 {i} 的记录不是有效的 JSON 对象 (字典)。", gr.update(visible=True)
            if 'content' not in record:
                # 后端 AdminCore 允许 content 为空但有 embedding，这里只警告
                print(f"警告: 批量插入数据中索引 {i} 的记录缺少 'content' 字段。", file=sys.stderr)
            if 'tags' not in record:
                return f"错误: 批量插入数据中索引 {i} 的记录缺少 'tags' 字段。", gr.update(visible=True)
            if not isinstance(record.get('tags'), list):
                return f"错误: 批量插入数据中索引 {i} 的记录的 'tags' 字段必须是 JSON 数组。", gr.update(visible=True)

            # 验证可选的 embedding 字段格式
            embedding = record.get('embedding')
            if embedding is not None:
                if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                     return f"错误: 批量插入数据中索引 {i} 的记录的 'embedding' 字段格式不正确 (应为 JSON 浮点数列表)。", gr.update(visible=True)
                # 前端不强制验证 embedding 维度，依赖后端检查和处理


        # 构建发送给后端的 payload
        json_payload = {"collection_name": collection_name, "records": records_to_insert}
        # 注意：批量插入的 auto_generate_embedding 和 batch_size 参数由后端 AdminCore 控制，
        # 此处 Gradio UI 不提供输入框。如果需要，可以在这里添加到 json_payload 中。
        # 例如: json_payload["auto_generate_embedding"] = True # 或从 UI 输入获取
        # json_payload["batch_size"] = 100 # 或从 UI 输入获取

        # 调用批量插入 API
        url = f"{ADMIN_FASTAPI_URL}/records/insert_many"
        try:
            print(f"调用 API: POST {url}，批量插入 {len(records_to_insert)} 条记录")
            response = requests.post(url, json=json_payload)
            response.raise_for_status() # 检查 HTTP 状态码

            # 尝试解析 JSON 响应并显示
            try:
                return display_api_result(response.json()), gr.update(visible=True)
            except json.JSONDecodeError:
                 return {"status": "error", "message": f"API 返回非 JSON 响应 (状态码: {response.status_code})。响应文本: {response.text}"}, gr.update(visible=True)

        except requests.exceptions.RequestException as e:
            # 处理 requests 库级别的错误，包括 HTTPError, ConnectionError, Timeout 等
            error_msg = f"API 请求失败: {e}"
            print(f"批量插入请求异常: {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # 尝试从错误响应中获取详细信息
            try:
                error_detail = e.response.json().get('detail', e.response.text)
                error_msg = f"批量插入 API 请求失败: HTTP 错误 - {e.response.status_code} {e.response.reason}。详情: {error_detail}"
            except (json.JSONDecodeError, AttributeError):
                pass # 如果响应不是 JSON 或没有 .json() 方法

            return {"status": "error", "message": error_msg}, gr.update(visible=True)
        except Exception as e:
            # 处理其他意外错误
            error_msg = f"批量插入过程中发生意外错误: {str(e)}"
            print(f"批量插入意外错误: {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": error_msg}, gr.update(visible=True)


    # --- 绑定 Gradio 事件 ---

    # 使用 .click() 方法将按钮点击事件绑定到相应的处理函数


    # 集合管理按钮绑定
    create_collection_btn.click(
        fn=handle_create_collection,
        inputs=[collection_name_input, custom_schema_input], # 增加自定义 schema 输入
        outputs=[results_output] # 只更新结果输出框
    )

    drop_collection_btn.click(
        fn=handle_drop_collection,
        inputs=[collection_name_input],
        outputs=[results_output]
    )

    list_collections_btn.click(
        fn=handle_list_collections,
        inputs=[],
        # 更新结果输出框和 Collection 列表输出框
        outputs=[results_output, collection_list_output]
    )

    has_collection_btn.click(
        fn=handle_has_collection,
        inputs=[collection_name_input],
        outputs=[results_output]
    )

    # 初始化 Collection 结构按钮绑定，包含所有高级参数
    init_collection_btn.click(
        fn=handle_init_collection,
        inputs=[
            collection_name_input,
            init_collection_drop_checkbox,
            custom_schema_input, # 使用与创建 Collection 相同的 custom_schema 输入框
            index_field_name_input,
            index_params_input
        ],
        outputs=[results_output]
    )

    create_index_btn.click(
        fn=handle_create_index,
        inputs=[collection_name_input, index_field_name_input, index_params_input],
        outputs=[results_output]
    )

    load_collection_btn.click(
        fn=handle_load_collection,
        inputs=[collection_name_input],
        outputs=[results_output]
    )


    # 记录插入按钮绑定
    insert_one_btn.click(
        fn=handle_insert_one,
        # inputs 包含 Collection 名称、content、tags (JSON string)、embedding (JSON string)
        inputs=[insert_collection_name_input, record_content_input, record_tags_input, record_embedding_input],
        outputs=[results_output]
    )

    insert_many_btn.click(
        fn=handle_insert_many,
        # inputs 包含 Collection 名称、文件对象、JSON 文本框内容
        inputs=[insert_collection_name_input, bulk_insert_file, bulk_insert_json_text],
        outputs=[results_output]
    )

    # 界面加载时自动执行一次列出 Collections 操作
    # 注意：demo.load 会在每次 UI 实例创建时运行，而不是用户每次刷新页面
    demo.load(
        fn=handle_list_collections,
        inputs=[],
        outputs=[results_output, collection_list_output]
    )

# pylint: enable=no-member # 重新启用警告

# --- 运行 Gradio 应用 ---

if __name__ == "__main__":
    # 确保 AdminCore.py 包含必要的 Milvus 连接设置或默认值
    print(f"NaviSearch Admin Gradio UI 正在启动，访问地址: http://127.0.0.1:7861")
    print(f"请确保您的 Admin FastAPI 服务正在运行在 {ADMIN_FASTAPI_URL}")
    # 启动 Gradio 应用，指定服务器端口
    demo.launch(server_port=7861)