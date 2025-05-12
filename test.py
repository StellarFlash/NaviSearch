# test.py - 使用 ORM 获取 Schema，用 MilvusClient 尝试 upsert

import traceback
import sys # 导入 sys 用于错误输出
import time # 导入 time 模块 (虽然 describe_collection 移除了，但保留导入以防万一)
from typing import List, Dict, Optional, Any

# 导入 Milvus 相关的类和函数
from pymilvus import MilvusClient, Collection, utility, Connections, DataType, FieldSchema, CollectionSchema

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

# --- 建立 Milvus ORM 连接 (用于 ORM 操作，如获取 Schema) ---
# connect 是幂等的，重复调用不会有问题。
# ORM 连接必须在 Collection() 或 utility() 调用前建立
try:
    Connections().connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, token=MILVUS_TOKEN)
    print("Milvus ORM connection established successfully with alias 'default'.")
    orm_connected = True
except Exception as e:
    print(f"Failed to establish Milvus ORM connection: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("Schema retrieval via ORM may not be possible.", file=sys.stderr)
    orm_connected = False


# --- 建立 MilvusClient 连接 (用于 Client 操作，如 upsert) ---
client = None # 初始化为 None
try:
    client = MilvusClient(token=MILVUS_TOKEN, host=MILVUS_HOST, port=MILVUS_PORT)
    print("MilvusClient instance created successfully.")
    client_created = True
except Exception as e:
    print(f"Failed to create MilvusClient instance: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("Cannot perform upsert operation without a working MilvusClient.", file=sys.stderr)
    client_created = False
    # 如果 MilvusClient 创建失败，继续执行，因为 ORM 可能还在工作

print("-" * 20)

# --- 检查 Collection 是否存在 (使用 MilvusClient 或 ORM) ---
collection_exists = False
if client_created and client.has_collection(collection_name):
    collection_exists = True
    print(f"Collection '{collection_name}' exists (checked by MilvusClient).")
elif orm_connected and utility.has_collection(collection_name, using='default'):
    collection_exists = True
    print(f"Collection '{collection_name}' exists (checked by ORM utility).")
else:
     print(f"Collection '{collection_name}' does not exist.", file=sys.stderr)
     print("Please create the collection first using the Admin API /collections/init or /collections/create.", file=sys.stderr)


if collection_exists:
    # --- 获取并打印 Collection 的 Schema (使用 ORM) ---
    # 我们信任 ORM 方式能够获取正确的 Schema
    if orm_connected:
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

                 if primary_field and getattr(primary_field, 'auto_id', False) is True: # 使用 getattr with default False for safety
                     print(f"\nConfirmed: Primary field '{primary_field.name}' has auto_id=True based on ORM schema.")

                     # --- 准备并 Upsert 测试记录 (使用 MilvusClient) ---
                     if client_created: # 确保 MilvusClient 成功创建
                         print(f"\nAttempting to upsert a test record into '{collection_name}' using MilvusClient...")
                         test_content = "这是一条用于测试 upsert 的记录。"
                         test_tags = ["测试", "upsert", "示例"]

                         try:
                             test_embedding = get_embedding(test_content)
                             if not isinstance(test_embedding, list) or len(test_embedding) != DEFAULT_DIM:
                                  raise ValueError(f"Generated embedding has invalid format or dimension: Expected list of size {DEFAULT_DIM}, got {len(test_embedding) if isinstance(test_embedding, list) else 'non-list'}.")

                             # 构建待插入的数据字典（不包含 id）
                             record_to_upsert = {
                                 "content": test_content,
                                 "tags": test_tags, # tags 对应 JSON 字段，直接放 Python list/dict 即可
                                 "embedding": test_embedding
                             }

                             # 执行 upsert 操作
                             # upsert 方法期望一个包含字典的列表
                             upsert_result = client.upsert(
                                 collection_name=collection_name,
                                 data=[record_to_upsert]
                             )

                             print("\nUpsert Result:")
                             print(upsert_result)

                             # 检查 upsert 结果
                             if upsert_result and upsert_result.get('insert_count', 0) > 0:
                                 print(f"\nSuccessfully upserted {upsert_result.get('insert_count', 0)} record(s).")
                                 if upsert_result.get('ids'):
                                      print(f"Inserted ID(s): {upsert_result['ids']}")
                             elif upsert_result:
                                 print("\nUpsert operation completed, but no new records were inserted or existing ones updated.")
                             else:
                                  print("\nUpsert operation result is unexpected or empty.")


                         except Exception as e:
                             print(f"Error during upsert operation with MilvusClient: {e}", file=sys.stderr)
                             traceback.print_exc(file=sys.stderr)
                             print("\nUpsert failed.")
                     else:
                         print("\nSkipping upsert because MilvusClient was not created successfully.", file=sys.stderr)


                 elif primary_field: # Primary field exists but auto_id is not True (based on ORM schema)
                      print(f"\nError: Primary field '{primary_field.name}' does NOT have auto_id=True ({getattr(primary_field, 'auto_id', 'N/A')}) based on ORM schema.", file=sys.stderr)
                      print("Cannot upsert data without providing 'id' when auto_id is False.", file=sys.stderr)
                 else: # Should not happen if ORM schema retrieval works and primary field exists
                      print("\nInternal Error: ORM schema retrieved, but could not find primary field marked as is_primary=True.", file=sys.stderr)

            else: # schema or schema.fields is empty or None
                 print("\nError: Failed to retrieve fields schema using ORM. 'schema.fields' was empty or None.", file=sys.stderr)
                 print("ORM Schema object:", schema) # Print schema object itself for debugging


        except Exception as e:
            # This catch block handles errors during ORM schema retrieval
            print(f"Error getting collection schema using ORM for '{collection_name}': {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("\nORM Schema retrieval failed.")
    else:
        print("\nSkipping schema retrieval and upsert because ORM connection failed.", file=sys.stderr)


# Script finishes regardless of upsert success/failure
print("-" * 20)
print("Script finished.")