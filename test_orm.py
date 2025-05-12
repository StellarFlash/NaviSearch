# test.py - 使用 ORM insert 尝试插入测试记录

import traceback
import sys
import time # 导入 time 模块 (虽然描述方法改了，但保留导入以防万一)
from typing import List, Dict, Optional, Any

# 导入 Milvus 相关的类和函数
from pymilvus import MilvusClient, Collection, utility, Connections, DataType, FieldSchema, CollectionSchema
# from pymilvus.orm.types import InsertRequest # 导入 InsertRequest 如果需要类型提示

# 请使用您的 Milvus 连接详情
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_TOKEN = "root:Milvus"
collection_name = "test_navi_search_collection" # 目标 Collection 名称

# --- 导入 get_embedding 函数，如果失败则使用模拟函数 ---
try:
    # 假设 utils.py 文件存在且包含 get_embedding 函数
    from utils import get_embedding
    print("Successfully imported get_embedding from utils.")
    # 根据 AdminCore 中的 DEFAULT_DIM 来确定嵌入维度
    try:
         from AdminCore import DEFAULT_DIM
    except ImportError:
         print("Warning: Could not import DEFAULT_DIM from AdminCore. Using fallback dimension 1024.", file=sys.stderr)
         DEFAULT_DIM = 1024 # Fallback dimension if AdminCore is not available

except ImportError:
    print("Warning: 'utils' module or 'get_embedding' not found. Using a dummy function and DEFAULT_DIM = 1024 for embedding generation.", file=sys.stderr)
    DEFAULT_DIM = 1024 # 使用默认维度
    # 模拟函数 for get_embedding if not available
    def get_embedding(text: str) -> List[float]:
        print(f"Using dummy get_embedding for text: '{text[:50]}...'")
        import random
        # 返回 DEFAULT_DIM 维度的随机向量
        return [random.random() for _ in range(DEFAULT_DIM)]

print(f"Attempting to establish Milvus connections...")

# --- 建立 Milvus ORM 连接 (用于 Schema 获取和 Insert) ---
# connect 是幂等的，重复调用不会有问题。
# ORM 连接必须在 Collection() 或 utility() 调用前建立
try:
    Connections().connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, token=MILVUS_TOKEN)
    print("Milvus ORM connection established successfully with alias 'default'.")
    orm_connected = True
except Exception as e:
    print(f"Failed to establish Milvus ORM connection: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("Schema retrieval and ORM insert may not be possible.", file=sys.stderr)
    orm_connected = False


# --- 建立 MilvusClient 连接 (仅用于检查 Collection 存在性，不用于 Insert) ---
client = None # 初始化为 None
client_created = False
if orm_connected: # 仅在 ORM 连接成功时尝试创建 MilvusClient (可选依赖)
    try:
        client = MilvusClient(token=MILVUS_TOKEN, host=MILVUS_HOST, port=MILVUS_PORT)
        print("MilvusClient instance created successfully.")
        client_created = True
    except Exception as e:
        print(f"Failed to create MilvusClient instance: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("MilvusClient is not available for existence check or other client operations.", file=sys.stderr)


print("-" * 20)

# --- 检查 Collection 存在性 (优先使用 MilvusClient，回退到 ORM utility) ---
collection_exists = False
if client_created and client.has_collection(collection_name):
    collection_exists = True
    print(f"Collection '{collection_name}' exists (checked by MilvusClient).")
elif orm_connected and utility.has_collection(collection_name, using='default'):
    collection_exists = True
    print(f"Collection '{collection_name}' exists (checked by ORM utility).")
else:
     print(f"Collection '{collection_name}' does not exist.", file=sys.stderr)
     if orm_connected:
         print("Please create the collection first using the Admin API /collections/init or /collections/create.", file=sys.stderr)
     else:
         print("Cannot check collection existence because no Milvus connection could be established.", file=sys.stderr)


if collection_exists and orm_connected: # 仅在 Collection 存在且 ORM 连接成功时进行 Schema 检查和 ORM Insert
    # --- 获取并打印 Collection 的 Schema (使用 ORM) ---
    try:
        print(f"Collection '{collection_name}' schema details (using ORM):")
        # 使用 ORM Collection 对象获取 Schema
        collection_orm = Collection(collection_name)
        # ORM schema 对象有 .fields 属性直接获取 FieldSchema 列表
        schema = collection_orm.schema

        print("\nFields in schema:")
        # 迭代 FieldSchema 对象列表
        if schema and schema.fields:
             for field in schema.fields:
                 # FieldSchema 对象有 .name, .dtype, .is_primary, .auto_id, .params 等属性
                 # 使用 getattr 确保兼容性
                 auto_id_val = getattr(field, 'auto_id', 'N/A (attribute not found)')
                 print(f"- Name: {field.name}, Type: {field.dtype}, Is Primary: {field.is_primary}, Auto ID: {auto_id_val}, Params: {field.params}")

             # 检查主键字段的 auto_id 是否为 True
             primary_field = next((f for f in schema.fields if f.is_primary), None)

             if primary_field and getattr(primary_field, 'auto_id', False) is True:
                 print(f"\nConfirmed: Primary field '{primary_field.name}' has auto_id=True based on ORM schema.")

                 # --- 准备并 Insert 测试记录 (使用 ORM Collection) ---
                 # ORM insert/upsert 需要 Collection 被加载
                 try:
                     print(f"Loading collection '{collection_name}' for ORM insert...")
                     # 在执行插入操作前加载 Collection
                     collection_orm.load()
                     print("Collection loaded.")

                     print(f"\nAttempting to insert a test record into '{collection_name}' using ORM...")
                     test_content = "这是一条用于测试 ORM insert 的记录。"
                     test_tags = ["测试", "ORM", "示例"]

                     try:
                         test_embedding = get_embedding(test_content)
                         if not isinstance(test_embedding, list) or len(test_embedding) != DEFAULT_DIM:
                              raise ValueError(f"Generated embedding has invalid format or dimension: Expected list of size {DEFAULT_DIM}, got {len(test_embedding) if isinstance(test_embedding, list) else 'non-list'}.")

                         # 构建待插入的数据字典（不包含 id）
                         record_to_insert = {
                             "content": test_content,
                             "tags": test_tags,
                             "embedding": test_embedding
                         }

                         # 执行 insert 操作 using ORM Collection object
                         # insert 方法期望一个包含字典的列表或 Entity 对象列表
                         # Auto-ID 处理由 ORM 基于 Schema 管理
                         # ORM insert 默认是 insert，如果想 upsert 需要用 collection_orm.upsert()
                         # 我们先尝试 insert
                         insert_result = collection_orm.insert([record_to_insert])

                         print("\nORM Insert Result:")
                         print(insert_result)

                         # ORM insert 结果通常返回一个 MutationWriteResult 对象
                         if hasattr(insert_result, 'insert_count') and insert_result.insert_count > 0:
                             print(f"\nSuccessfully inserted {insert_result.insert_count} record(s) using ORM.")
                             if hasattr(insert_result, 'primary_keys'):
                                  print(f"Inserted primary key(s): {insert_result.primary_keys}")
                             # 插入后需要 flush 才能立即可搜 (可选)
                             try:
                                 print("Flushing collection after ORM insert...")
                                 collection_orm.flush()
                                 print("Collection flushed.")
                             except Exception as flush_e:
                                  print(f"Warning: Failed to flush collection after ORM insert: {flush_e}", file=sys.stderr)
                                  traceback.print_exc(file=sys.stderr)

                         elif hasattr(insert_result, 'insert_count'):
                              print("\nORM insert operation completed, but no records were inserted.")
                         else:
                              print("\nORM insert operation result is unexpected or empty.")


                     except Exception as e:
                         print(f"Error during ORM insert operation: {e}", file=sys.stderr)
                         traceback.print_exc(file=sys.stderr)
                         print("\nORM Insert failed.")

                 except Exception as load_e:
                      print(f"Error loading collection '{collection_name}' for ORM insert: {load_e}", file=sys.stderr)
                      traceback.print_exc(file=sys.stderr)
                      print("\nSkipping ORM insert because collection could not be loaded.")


             elif primary_field: # Primary field exists but auto_id is not True (based on ORM schema)
                  print(f"\nError: Primary field '{primary_field.name}' does NOT have auto_id=True ({getattr(primary_field, 'auto_id', 'N/A')}) based on ORM schema.", file=sys.stderr)
                  print("Cannot insert data without providing 'id' when auto_id is False.", file=sys.stderr)
             else: # Should not happen if ORM schema retrieval works and primary field exists
                  print("\nInternal Error: ORM schema retrieved, but could not find primary field marked as is_primary=True.", file=sys.stderr)

        else: # schema or schema.fields is empty or None
             print("\nError: Failed to retrieve fields schema using ORM. 'schema.fields' was empty or None.", file=sys.stderr)
             print("ORM Schema object:", schema) # 打印 schema object itself for debugging

    except Exception as e:
        # This catch block handles errors during ORM schema retrieval
        print(f"Error getting collection schema using ORM for '{collection_name}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("\nORM Schema retrieval failed.")

else: # Collection does not exist OR ORM connection failed
    if not collection_exists:
        print(f"Collection '{collection_name}' does not exist.", file=sys.stderr)
    if not orm_connected:
         print("Cannot proceed with schema check or ORM insert without a successful ORM connection.", file=sys.stderr)


print("-" * 20)
print("Script finished.")