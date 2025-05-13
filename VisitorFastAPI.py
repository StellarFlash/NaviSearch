from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# 假设 VisitorCore.py 文件位于与此文件相同的目录下
try:
    from VisitorCore import NaviSearchVisitor, DEFAULT_MILVUS_HOST, DEFAULT_MILVUS_PORT, DEFAULT_MILVUS_TOKEN, DEFAULT_COLLECTION_NAME
    # 导入 VisitorCore.py 中可能需要的模型和函数
    # from utils import get_embedding, get_response, get_filter, flatten_nested_structure
    # from Search.SearchEngine import SearchEngine
    # from pymilvus import MilvusClient, FieldSchema, CollectionSchema, Collection, DataType, Connections
except ImportError as e:
    print(f"Error importing VisitorCore module or its dependencies: {e}")
    print("Please ensure VisitorCore.py and its required modules (Search, utils, pymilvus) are in the correct path.")
    # 如果导入失败，可以选择在这里退出或者在服务启动时抛出错误
    raise e

# --- 全局变量控制 IP 和端口 ---
SERVICE_HOST = "0.0.0.0"  # 监听所有可用接口
SERVICE_PORT = 8000      # 服务端口

# --- 全局 Visitor 实例 ---
# 在应用启动时创建 Visitor 实例
# 注意：这将使用 VisitorCore.py 中定义的默认 Milvus 连接参数
# 如果需要动态配置 Milvus 连接，可能需要在 /connect 接口中传递参数，
# 或者考虑将 Visitor 实例的创建延迟到 /connect 被调用时。
# 为了遵循用户要求使用全局变量控制服务IP/Port和封装Visitor，我们在这里创建全局Visitor。
# Milvus连接本身在 Visitor 实例内部管理。
try:
    global_visitor_instance = NaviSearchVisitor(
        milvus_host=DEFAULT_MILVUS_HOST,
        milvus_port=DEFAULT_MILVUS_PORT,
        milvus_token=DEFAULT_MILVUS_TOKEN,
        collection_name=DEFAULT_COLLECTION_NAME # 使用默认collection，可以通过set_collection接口更改
    )
    print("NaviSearchVisitor instance created globally.")
except Exception as e:
    print(f"Failed to create NaviSearchVisitor instance: {e}")
    global_visitor_instance = None # 如果创建失败，设置为 None

# --- FastAPI 应用实例 ---
app = FastAPI()

# --- 请求体模型定义 ---

class SetCollectionRequest(BaseModel):
    collection_name: str

class AddTagsRequest(BaseModel):
    tag_content_to_parse: str

class SearchRequest(BaseModel):
    query_text: str
    filter_tags: Optional[List[str]] = None
    mode: str = "standard" # "standard" or "agentic"
    retrieval_top_k: int = 20
    rerank_strategy: Optional[str] = "ranking" # maps to rerank mode like "ranking" or "filtering"
    # Parameters specific to search modes (optional, handled based on mode)
    rerank_top_k_standard: Optional[int] = None # for standard search
    rerank_stop_size_agentic: Optional[int] = None # for agentic search
    max_iterations_agentic: Optional[int] = None # for agentic search


# --- 辅助函数：检查 Visitor 实例是否可用 ---
def get_visitor_instance():
    """ Dependency to get the global visitor instance and check its availability. """
    if global_visitor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="NaviSearchVisitor instance failed to initialize."
        )
    # 检查 Milvus 连接状态，某些操作需要连接
    # Visitor 内部的 connect_milvus 和 search 已经包含了连接/可用性检查，
    # 这里可以不再重复检查，或者根据需要添加更细致的检查。
    return global_visitor_instance

# --- API 路由定义 ---

@app.get("/")
def read_root():
    """ Root endpoint to check if the service is running. """
    return {"message": "NaviSearch Visitor Service is running!"}

@app.post("/connect")
def connect_milvus_endpoint():
    """ Connects the Visitor instance to Milvus. """
    visitor = get_visitor_instance()
    try:
        result = visitor.connect_milvus()
        if result.get('status') == 'error':
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))
        return result
    except Exception as e:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during connection: {e}")


@app.post("/disconnect")
def disconnect_milvus_endpoint():
    """ Disconnects the Visitor instance from Milvus. """
    visitor = get_visitor_instance()
    try:
        result = visitor.disconnect_milvus()
        if result.get('status') == 'error':
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))
        return result
    except Exception as e:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during disconnection: {e}")

@app.post("/set_collection")
def set_collection_endpoint(request: SetCollectionRequest):
    """ Sets the target collection name for the Visitor instance. """
    visitor = get_visitor_instance()
    try:
        result = visitor.set_target_collection_name(request.collection_name)
        # 注意：set_target_collection_name 可能会返回 status='warning'
        if result.get('status') == 'error':
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get('message'))
        return result
    except Exception as e:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while setting collection: {e}")


@app.get("/get_collection")
def get_collection_endpoint():
    """ Gets the current target collection name of the Visitor instance. """
    visitor = get_visitor_instance()
    try:
        collection_name = visitor.get_current_target_collection_name()
        return {"current_collection": collection_name}
    except Exception as e:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while getting collection name: {e}")


@app.post("/add_tags")
def add_tags_endpoint(request: AddTagsRequest):
    """ Adds active tags for the current session. """
    visitor = get_visitor_instance()
    try:
        result = visitor.add_tags(request.tag_content_to_parse)
        # Note: add_tags can return status='warning' or 'info'
        if result.get('status') == 'error':
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get('message'))
        return result
    except Exception as e:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while adding tags: {e}")


@app.post("/clear_tags")
def clear_tags_endpoint():
    """ Clears all active tags for the current session. """
    visitor = get_visitor_instance()
    try:
        result = visitor.clear_tags()
        return result
    except Exception as e:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while clearing tags: {e}")


@app.get("/get_tags")
def get_tags_endpoint():
    """ Gets the current list of active tags. """
    visitor = get_visitor_instance()
    try:
        active_tags = visitor.get_active_tags()
        return {"active_tags": active_tags}
    except Exception as e:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while getting tags: {e}")


@app.post("/search")
def search_endpoint(request: SearchRequest):
    """ Performs a search using the Visitor instance. """
    visitor = get_visitor_instance()

    # Prepare kwargs based on the selected mode
    search_kwargs = {}
    if request.mode == "standard":
        if request.rerank_top_k_standard is not None:
            search_kwargs['rerank_top_k'] = request.rerank_top_k_standard
    elif request.mode == "agentic":
        if request.rerank_stop_size_agentic is not None:
            search_kwargs['rerank_stop_size'] = request.rerank_stop_size_agentic
        if request.max_iterations_agentic is not None:
            search_kwargs['max_iterations'] = request.max_iterations_agentic
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported search mode: {request.mode}. Supported modes are 'standard' and 'agentic'."
        )

    print(request.query_text)
    print(request.filter_tags)
    print(request.mode)
    print(request.retrieval_top_k)
    print(request.rerank_strategy)
    print(search_kwargs)
    try:
        # Check if Milvus is connected before searching
        if not visitor._is_connected:
             # Attempt to connect automatically if not connected (optional, depends on desired behavior)
             # Or just raise an error requiring the client to call /connect first.
             # Let's require the client to connect first for clearer state management.
            visitor.connect_milvus()  # This will raise an error if connection fails, which will be caught by FastAP
        print(request.query_text)
        print(request.filter_tags)
        print(request.mode)
        print(request.retrieval_top_k)
        print(request.rerank_strategy)
        print(search_kwargs)
        # Perform the search
        search_result = visitor.search(
            query_text=request.query_text,
            filter_tags=request.filter_tags, # Use filter_tags from request body if provided, otherwise Visitor uses active_tags
            mode=request.mode,
            retrieval_top_k=request.retrieval_top_k,
            rerank_strategy=request.rerank_strategy,
            **search_kwargs # Pass mode-specific parameters
        )
        print(search_result)
        if search_result.get('status') == 'error':
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=search_result.get('message'))
        elif search_result.get('status') == 'fail': # Agentic search might 'fail' to converge
             # Depending on desired behavior, this could be 200 with a 'fail' status in body, or a specific error code
             # Let's return 200 but include the status and message from the visitor result
             return search_result

        return search_result

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions
        raise http_exc
    except Exception as e:
        import traceback
        traceback.print_exc() # Print traceback for debugging server side
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during search: {str(e)}")

# --- 运行服务 ---
if __name__ == "__main__":
    import uvicorn
    # 使用全局变量 SERVICE_HOST 和 SERVICE_PORT 运行服务
    print(f"Starting NaviSearch Visitor Service on http://{SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)