# AdminFastAPI.py
# 提供 Milvus Collection 和 Document 管理的 FastAPI 接口

import traceback
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field, ValidationError
import sys

# 从 pymilvus 导入构建 schema 所需的类和枚举
from pymilvus import FieldSchema, CollectionSchema, DataType

# 尝试导入 AdminCore 和 get_embedding
try:
    # 假设 AdminCore.py 在同一目录下
    from AdminCore import NaviSearchAdmin, DEFAULT_DIM
except ImportError as e:
    print(f"错误：无法从 AdminCore.py 导入 NaviSearchAdmin：{e}", file=sys.stderr)
    print("请确保 AdminCore.py 文件存在且位于正确的路径。", file=sys.stderr)
    sys.exit(1) # 如果核心 Admin 功能不可用，则退出

# 尝试从 utils 导入 get_embedding，如果失败则使用一个模拟函数
try:
    from utils import get_embedding
    print("成功从 utils 导入 get_embedding。")
except ImportError:
    print("警告：未找到 'utils' 模块或 'get_embedding' 函数。将使用模拟函数生成 embedding。", file=sys.stderr)
    # 如果 get_embedding 不可用，提供一个模拟函数
    def get_embedding(text: str) -> List[float]:
        print(f"使用模拟函数生成 embedding，输入文本（前50字符）：'{text[:50]}...'")
        import random
        # 确保模拟 embedding 的维度与默认维度一致
        return [random.random() for _ in range(DEFAULT_DIM)]


# --- API 请求体的 Pydantic 模型 ---

# 将 API 请求中使用的字符串数据类型映射到 Pymilvus 的 DataType 枚举
dtype_map = {
    "BOOL": DataType.BOOL,
    "INT8": DataType.INT8,
    "INT16": DataType.INT16,
    "INT32": DataType.INT32,
    "INT64": DataType.INT64,
    "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.DOUBLE,
    "VARCHAR": DataType.VARCHAR,
    "JSON": DataType.JSON,
    "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "BINARY_VECTOR": DataType.BINARY_VECTOR,
}

# 用于定义 Collection 字段的 Pydantic 模型
class FieldSchemaModel(BaseModel):
    name: str = Field(..., description="字段名称")
    dtype: str = Field(..., description=f"数据类型，可选值：{list(dtype_map.keys())}")
    is_primary: bool = Field(False, description="是否为主键")
    auto_id: bool = Field(False, description="是否自动生成ID (仅对主键 INT64 有效)")
    description: Optional[str] = Field(None, description="字段描述")
    max_length: Optional[int] = Field(None, description="VARCHAR 类型的最大长度")
    dim: Optional[int] = Field(None, description="向量类型的维度")
    params: Optional[Dict] = Field(None, description="其他字段特定参数")

# 用于定义整个 Collection Schema 的 Pydantic 模型
class CollectionSchemaModel(BaseModel):
    fields: List[FieldSchemaModel] = Field(..., description="字段定义列表")
    description: Optional[str] = Field(None, description="Collection 描述")
    enable_dynamic_field: bool = Field(False, description="是否启用动态字段")


def convert_field_schema_model_to_pymilvus(field_model: FieldSchemaModel) -> FieldSchema:
    """将 Pydantic 字段模型转换为 Pymilvus 字段模型。"""
    dtype_enum = dtype_map.get(field_model.dtype.upper())
    if dtype_enum is None:
        raise ValueError(f"未知的数据类型: {field_model.dtype}")

    # 强制检查 VARCHAR 字段必须提供 max_length
    if dtype_enum == DataType.VARCHAR and field_model.max_length is None:
        raise ValueError(f"VARCHAR 字段 '{field_model.name}' 必须提供 max_length 参数")

    # 强制检查向量字段必须提供 dim
    if dtype_enum in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR] and field_model.dim is None:
        raise ValueError(f"向量字段 '{field_model.name}' 必须提供 dim 参数")

    field_params = {
        "name": field_model.name,
        "dtype": dtype_enum,
        "is_primary": field_model.is_primary,
        "auto_id": field_model.auto_id,
    }

    # 添加描述（如果有）
    if field_model.description:
        field_params["description"] = field_model.description

    # 添加类型特定参数
    if dtype_enum == DataType.VARCHAR:
        field_params["max_length"] = field_model.max_length
    elif dtype_enum in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
        field_params["dim"] = field_model.dim

    # 添加额外参数（如果有）
    if field_model.params:
        field_params.update(field_model.params)

    return FieldSchema(**field_params)

# 辅助函数：将 Pydantic CollectionSchemaModel 转换为 Pymilvus CollectionSchema
def convert_collection_schema_model_to_pymilvus(schema_model: CollectionSchemaModel) -> CollectionSchema:
    """将 Pydantic Collection Schema 模型转换为 Pymilvus Collection Schema。"""
    fields = [convert_field_schema_model_to_pymilvus(f) for f in schema_model.fields]
    return CollectionSchema(
        fields=fields,
        description=schema_model.description or "",
        enable_dynamic_field=schema_model.enable_dynamic_field
    )


class RecordDataModel(BaseModel):
    """
    支持动态字段的记录数据模型，同时强制要求核心字段。
    会根据 Collection schema 动态验证字段类型。
    """
    # 强制要求的核心字段
    content: str = Field(..., description="文本内容")
    tags: Dict[str, Any] = Field(default_factory=dict, description="标签字典 (JSON 格式)")
    embedding: Optional[List[float]] = Field(None, description="向量嵌入 (可选)")

    # 允许动态字段 (兼容 Pydantic V2)
    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "content": "文本内容",
                    "tags": {"category": "news"},
                    "title": "可选标题字段",
                    "source": "可选来源字段",
                    "custom_field": "自定义字段值"
                }
            ]
        }
    }

class InsertRecordRequest(BaseModel):
    collection_name: str = Field(..., description="目标集合名称")
    record_data: RecordDataModel = Field(..., description="记录数据")
    auto_generate_embedding: bool = Field(
        True,
        description="是否自动生成缺失的 embedding"
    )
    strict_validation: bool = Field(
        False,
        description="是否严格验证字段类型 (根据集合 schema)"
    )

class InsertRecordsRequest(BaseModel):
    collection_name: str = Field(..., description="目标集合名称")
    records: List[RecordDataModel] = Field(..., description="记录列表")
    auto_generate_embedding: bool = Field(True, description="自动生成 embedding")
    batch_size: int = Field(100, ge=1, le=1000, description="批量插入大小")
    strict_validation: bool = Field(False, description="严格字段类型验证")


# 初始化 Collection 结构的请求体模型
class InitCollectionRequest(BaseModel):
    collection_name: str = Field(..., description="Collection 名称")
    drop_existing: bool = Field(False, description="如果 Collection 已存在是否先删除")
    custom_schema: Optional[CollectionSchemaModel] = Field(None, description="自定义 Collection Schema (可选，如果未提供则使用 Admin 默认 Schema)")
    index_params: Optional[Dict] = Field(None, description="向量字段的索引参数字典 (可选，如果未提供则使用 Admin 默认参数，如 AUTOINDEX, COSINE)")
    vector_field_name: str = Field("embedding", description="要创建索引的向量字段名") # 允许指定向量字段名


# 创建 Collection 的请求体模型
class CreateCollectionRequest(BaseModel):
    collection_name: str = Field(..., description="Collection 名称")
    custom_schema: Optional[CollectionSchemaModel] = Field(None, description="自定义 Collection Schema (可选，如果未提供则使用 Admin 默认 Schema)")


# 加载 Collection 的请求体模型
class LoadCollectionRequest(BaseModel):
    collection_name: str = Field(..., description="Collection 名称")

# 创建索引的请求体模型
class CreateIndexRequest(BaseModel):
    collection_name: str = Field(..., description="Collection 名称")
    field_name: str = Field("embedding", description="要创建索引的字段名称，默认为 'embedding'")
    index_params: Optional[Dict] = Field(None, description="索引参数字典 (可选)")


# --- FastAPI 应用设置 ---

app = FastAPI(
    title="NaviSearch Milvus Admin API",
    description="提供 Milvus Collection 和 Document 管理功能，支持自定义 Schema 和自动生成 Embedding",
    version="1.1.0", # 更新版本号以反映新功能
)

# 依赖注入：获取 NaviSearchAdmin 实例
def get_admin() -> NaviSearchAdmin:
    """提供 NaviSearchAdmin 的依赖"""
    # 可以在此处从环境变量或配置文件加载 Milvus 连接信息
    # 这里使用 AdminCore 中的默认值
    # TODO: 考虑通过环境变量或配置文件使连接详情可配置
    return NaviSearchAdmin()

# --- API 端点 ---

@app.post("/collections/init", summary="初始化 Milvus Collection 结构 (创建 Collection 和索引)")
def api_init_collection(request: InitCollectionRequest, admin: NaviSearchAdmin = Depends(get_admin)):
    """
    高级功能：一步到位初始化 Collection 结构，包括创建 Collection 和创建索引。
    如果指定 `drop_existing` 为 True，则先删除已存在的同名 Collection。
    支持通过 `custom_schema` 自定义 Collection 结构，并通过 `index_params` 自定义索引参数。
    `vector_field_name` 用于指定在哪个向量字段上创建索引。
    """
    try:
        # 将 Pydantic schema 模型转换为 Pymilvus CollectionSchema 对象
        pymilvus_schema = None
        if request.custom_schema:
            pymilvus_schema = convert_collection_schema_model_to_pymilvus(request.custom_schema)

        result = admin.init_collection_structure(
            collection_name=request.collection_name,
            drop_existing=request.drop_existing,
            custom_schema=pymilvus_schema,
            index_params=request.index_params,
            vector_field_name=request.vector_field_name
        )
        if result.get('status') == 'error':
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))
        return result
    except ValueError as ve:
        # 捕获 schema 转换中的验证错误
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Schema 验证错误: {ve}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部服务器错误: {e}")


@app.post("/collections/index", summary="为 Collection 创建索引")
def api_create_index(request: CreateIndexRequest, admin: NaviSearchAdmin = Depends(get_admin)):
    """
    为指定 Collection 的指定字段创建索引 (通常是 embedding 字段)。
    可以提供可选的 `index_params` 来定制索引类型和参数。
    """
    try:
        result = admin.create_collection_index(request.collection_name, request.field_name, request.index_params)
        # 如果索引已经存在，不返回 500 错误，而是返回 info 状态
        if result.get('status') == 'error' and 'already exists' not in result.get('message', ''):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部服务器错误: {e}")


@app.post("/collections/load", summary="将 Collection 加载到内存")
def api_load_collection(request: LoadCollectionRequest, admin: NaviSearchAdmin = Depends(get_admin)):
    """
    将指定 Collection 加载到 Milvus 内存中，以备搜索。
    加载是进行向量搜索的前提。
    """
    try:
        result = admin.load_collection_to_memory(request.collection_name)
        if result.get('status') == 'error':
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部服务器错误: {e}")


@app.post("/collections/create", summary="创建新的 Collection")
def api_create_collection(request: CreateCollectionRequest, admin: NaviSearchAdmin = Depends(get_admin)):
    """
    创建一个新的 collection (仅结构，不加载数据或索引)。
    可以通过 `custom_schema` 参数自定义 Collection 结构，如果未提供则使用 Admin 默认 Schema。
    """
    try:
        # 将 Pydantic schema 模型转换为 Pymilvus CollectionSchema 对象
        pymilvus_schema = None
        if request.custom_schema:
            pymilvus_schema = convert_collection_schema_model_to_pymilvus(request.custom_schema)

        result = admin.create_collection(
            collection_name=request.collection_name,
            custom_schema=pymilvus_schema
        )
        # 如果 Collection 已经存在，不返回 500 错误，而是返回 info 状态
        if result.get('status') == 'error' and 'already exists' not in result.get('message', ''):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))
        return result
    except ValueError as ve:
         # 捕获 schema 转换中的验证错误
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Schema 验证错误: {ve}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部服务器错误: {e}")


@app.delete("/collections/{collection_name}", summary="删除指定的 Collection")
def api_drop_collection(collection_name: str, admin: NaviSearchAdmin = Depends(get_admin)):
    """
    删除指定的 collection。
    """
    if not collection_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Collection 名称不能为空。")
    try:
        result = admin.drop_collection(collection_name)
        # 如果 Collection 不存在，不返回 500 错误，而是返回 info 状态
        if result.get('status') == 'error' and 'does not exist' not in result.get('message', ''):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部服务器错误: {e}")


@app.get("/collections", summary="列出所有 Collection")
def api_list_collections(admin: NaviSearchAdmin = Depends(get_admin)):
    """
    列出 Milvus 中所有的 collection 名称。
    """
    try:
        result = admin.list_collections()
        if result.get('status') == 'error':
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部服务器错误: {e}")

@app.get("/collections/{collection_name}/exists", summary="检查 Collection 是否存在")
def api_has_collection(collection_name: str, admin: NaviSearchAdmin = Depends(get_admin)):
    """
    检查 Milvus 中是否存在指定的 collection。
    返回一个布尔值表示是否存在。
    """
    if not collection_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Collection 名称不能为空。")
    try:
        exists = admin.has_collection(collection_name)
        return {'collection_name': collection_name, 'exists': exists}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部服务器错误: {e}")


@app.post("/records/insert_one", summary="插入单条记录")
def api_insert_record(request: InsertRecordRequest, admin: NaviSearchAdmin = Depends(get_admin)):
    """
    插入一条记录到指定的 collection。
    记录数据必须包含 'content' 和 'tags' (JSON 类型)。
    'embedding' (List[float]) 字段是可选的，如果未提供且 `auto_generate_embedding` 为 True，将尝试使用 'content' 字段自动生成。
    可以包含符合 Collection schema 或动态字段的其他任意字段。
    """
    # 使用 model_dump() 或 dict() 将 Pydantic 模型转换为字典，以便传递给 AdminCore
    try:
        record_data_dict = request.record_data.model_dump() # Pydantic V2+
    except AttributeError:
        record_data_dict = request.record_data.dict() # Pydantic V1

    try:
        result = admin.insert_record(
            collection_name=request.collection_name,
            record_data=record_data_dict,
            auto_generate_embedding=request.auto_generate_embedding # 传递自动生成 embedding 的标志
        )
        if result.get('status') == 'error':
             # 如果错误是 Collection 不存在导致，返回 404
            if "does not exist" in result.get('message', ''):
                 raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result.get('message'))
            # 其他错误返回 500
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"插入单条记录时发生内部服务器错误: {e}")


@app.post("/records/insert_many", summary="批量插入记录")
def api_insert_records(request: InsertRecordsRequest, admin: NaviSearchAdmin = Depends(get_admin)):
    """
    批量插入多条记录到指定的 collection。
    每条记录必须包含 'content' 和 'tags' (JSON 类型)。
    'embedding' (List[float]) 字段是可选的，如果未提供且 `auto_generate_embedding` 为 True，将尝试使用 'content' 字段自动生成。
    记录可以包含符合 Collection schema 或动态字段的其他任意字段。
    支持通过 `batch_size` 控制分批插入大小。
    """
    # 将 Pydantic 模型列表转换为字典列表，保留额外字段
    records_list = []
    for record in request.records:
        try:
            records_list.append(record.model_dump()) # Pydantic V2+
        except AttributeError:
            records_list.append(record.dict()) # Pydantic V1

    if not records_list:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体中未包含要插入的记录。")

    try:
        result = admin.insert_records(
            collection_name=request.collection_name,
            records=records_list,
            auto_generate_embedding=request.auto_generate_embedding, # 传递自动生成 embedding 的标志
            batch_size=request.batch_size # 传递批量大小
        )
        if result.get('status') == 'error':
            # 如果错误是 Collection 不存在导致，返回 404
            if "does not exist" in result.get('message', ''):
                 raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result.get('message'))
            # 其他错误返回 500
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get('message'))

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"批量插入记录时发生内部服务器错误: {e}")


# --- 运行 FastAPI 应用 ---

if __name__ == "__main__":
    import uvicorn
    # 确保 AdminCore.py 包含必要的 Milvus 连接设置或默认值
    print("FastAPI 服务器正在启动，监听地址: http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)