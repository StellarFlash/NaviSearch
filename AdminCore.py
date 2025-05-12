# AdminCore.py

import traceback
import sys
from typing import List, Dict, Optional, Any
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, Collection, DataType, Connections, utility
# 导入 ORM Insert 结果类型以便在处理结果时使用 getattr
from utils import get_embedding, flatten_nested_structure

DEFAULT_DIM = 1024
DEFAULT_MILVUS_TOKEN = "root:Milvus"
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"

class NaviSearchAdmin:
     """
     负责 Milvus Collection 的管理和 Document 的插入。
     该类设计为无状态工具类，专注 Milvus 原生操作（使用 ORM 进行插入，Client 进行其他管理）。
     数据预处理（如 embedding 生成）由外部负责。
     """
     def __init__(self,
                    dim: int = DEFAULT_DIM,
                    milvus_token: str = DEFAULT_MILVUS_TOKEN,
                    milvus_host: str = DEFAULT_MILVUS_HOST,
                    milvus_port: str = DEFAULT_MILVUS_PORT):
          """
          初始化 Milvus Admin 工具类。
          """
          self.dim = dim
          self.milvus_token = milvus_token
          self.milvus_host = milvus_host
          self.milvus_port = milvus_port

          # 使用 MilvusClient 进行 Collection 级别的管理操作 (has_collection, list_collections, drop_collection, create_collection)
          # MilvusClient 使用独立的连接配置
          try:
               self.client = MilvusClient(token=self.milvus_token, host=self.milvus_host, port=self.milvus_port)
               print("MilvusClient initialized successfully.")
          except Exception as e:
               print(f"Failed to initialize MilvusClient: {e}")
               traceback.print_exc()
               self.client = None # 如果失败，将 client 置为 None


          # 定义 Collection 的默认字段和 Schema
          # 这个 self.schema 将作为未指定自定义 schema 时的默认选项
          self.default_fields = [
               FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
               FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
               FieldSchema(name="tags", dtype=DataType.JSON), # 使用 JSON 类型存储标签
               FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim) # 默认 embedding 字段使用 self.dim
          ]
          self.schema = CollectionSchema(self.default_fields, description=f"Default NaviSearch schema with {self.dim} dim embedding")

          # 连接 Pymilvus ORM 接口，用于索引创建、Collection 加载以及数据插入 (使用 ORM 进行插入)
          # Collection(name) 需要依赖 Connections() 中注册的连接别名 (默认为 'default')
          self._connect_milvus_orm()


     def _connect_milvus_orm(self):
          """
          连接Pymilvus ORM接口。用于索引创建、Collection加载、数据插入等操作。
          直接尝试连接，依赖 Connections().connect 的幂等性。
          """
          try:
               # 尝试建立或获取已有的 ORM 连接
               Connections().connect(alias="default", host=self.milvus_host, port=self.milvus_port, token=self.milvus_token)
               print(f"Attempted to connect Milvus ORM interface with alias 'default' at {self.milvus_host}:{self.milvus_port}")
          except Exception as e:
               print(f"Failed to connect to Milvus ORM interface: {e}")
               traceback.print_exc()
               # ORM 连接失败可能导致依赖 ORM 的方法无法工作，但 MilvusClient 可能仍然可用


     # --- Collection Management Methods (mostly using MilvusClient, Schema using ORM) ---

     # Methods like init_collection_structure, create_collection_index, load_collection_to_memory,
     # create_collection, drop_collection, list_collections, has_collection
     # These methods can largely remain as they were, using self.client or ORM Collection/utility

     # Example: has_collection using MilvusClient
     def has_collection(self, collection_name: str) -> bool:
          """
          检查是否存在指定的 collection。
          使用 MilvusClient 进行检查。
          """
          if not collection_name or not self.client:
               return False
          try:
               return self.client.has_collection(collection_name)
          except Exception as e:
               print(f"Error checking if collection '{collection_name}' exists using MilvusClient: {e}")
               traceback.print_exc()
               # 如果 MilvusClient 检查失败，尝试使用 ORM utility 作为回退
               if Connections().has_connection(alias="default"):
                    try:
                         print(f"Attempting ORM utility check for collection '{collection_name}'...")
                         return utility.has_collection(collection_name, using='default')
                    except Exception as orm_e:
                         print(f"Error checking if collection '{collection_name}' exists using ORM utility: {orm_e}")
                         traceback.print_exc()
               # 如果两种方式都失败，默认认为不存在
               return False

     def create_collection(self, collection_name: str, custom_schema: Optional[CollectionSchema] = None) -> Dict[str, Any]:
          """
          创建一个新的 collection (仅结构，不加载数据或索引)。
          如果提供了 custom_schema，则使用它；否则，使用 Admin 类初始化时定义的默认 self.schema。
          使用 MilvusClient 进行创建。

          Args:
               collection_name (str): 要创建的 Collection 名称。
               custom_schema (Optional[CollectionSchema]): 用户提供的自定义 CollectionSchema。默认为 None。

          Returns:
               Dict[str, Any]: 操作结果。
          """
          if not collection_name or not self.client:
               return {'status': 'error', 'message': "Collection name cannot be empty or MilvusClient not initialized."}

          try:
               if self.client.has_collection(collection_name):  # Use MilvusClient check
                    message = f"Collection '{collection_name}' already exists."
                    print(message)
                    return {'status': 'info', 'message': message}
               else:
                    # 确定要使用的 schema
                    schema_to_use = custom_schema if custom_schema is not None else self.schema

                    # 确保 schema_to_use 是一个有效的 CollectionSchema 对象
                    if not isinstance(schema_to_use, CollectionSchema):
                         error_msg = "Invalid schema provided. Must be an instance of pymilvus.CollectionSchema."
                         print(error_msg)
                         return {'status': 'error', 'message': error_msg}

                    # 检查 schema 是否至少包含一个向量字段，这是 Milvus 的典型要求
                    has_vector_field = False
                    if schema_to_use.fields:
                         for field in schema_to_use.fields:
                              if field.dtype == DataType.FLOAT_VECTOR or field.dtype == DataType.BINARY_VECTOR:
                                   has_vector_field = True
                                   # 同时可以获取自定义 schema 的 dim，如果只有一个向量字段
                                   # 如果有多个向量字段，则 self.dim 的概念可能需要重新考虑或针对特定字段
                                   # 对于默认 schema，self.dim 是明确的
                              if schema_to_use == self.schema: # 如果是默认 schema
                                   print(f"Using default schema with embedding dimension: {self.dim}")
                              # else: # 如果是自定义 schema
                                   # custom_dim = field.params.get('dim') if field.params else 'unknown'
                                   # print(f"Using custom schema with embedding dimension: {custom_dim} for field '{field.name}'")
                                   break

                    if not has_vector_field and schema_to_use.enable_dynamic_field != True: # 动态字段的schema可能不立即声明向量字段
                         print(f"Warning: The provided schema for '{collection_name}' does not explicitly contain a vector field. Ensure this is intended.")

                         print(f"Attempting to create collection '{collection_name}' with {'custom' if custom_schema else 'default'} schema.")
                    print(schema_to_use)
                    self.client.create_collection(collection_name, schema=schema_to_use)
                    message = f"Collection '{collection_name}' created successfully."
                    print(message)
                    return {'status': 'success', 'message': message}
          except Exception as e:
               error_msg = f"Error creating collection '{collection_name}' using MilvusClient: {e}"
               print(error_msg)
               traceback.print_exc()
               return {'status': 'error', 'message': error_msg}


     def insert_record(self, collection_name: str, record_data: Dict, auto_generate_embedding: bool = True) -> Dict[str, Any]:
        """
        插入记录到指定collection，具有以下特性：
        1. 强制检查content和tags字段
        2. **将tags字段展开为List[str]**
        3. 如果embedding字段缺失且auto_generate_embedding为True，则自动调用get_embedding生成
        4. 允许其他任意字段（支持动态schema）

        参数:
            collection_name: 集合名称
            record_data: 要插入的数据字典
            auto_generate_embedding: 当embedding缺失时是否自动生成

        返回:
            操作结果字典
        """
        # 基础验证
        if not collection_name:
            return {'status': 'error', 'message': "Collection name cannot be empty."}
        if not isinstance(record_data, dict):
            return {'status': 'error', 'message': "Invalid record_data format. Must be a dictionary."}

        # 强制检查content和tags字段
        required_fields = ['content', 'tags']
        missing_fields = [field for field in required_fields if field not in record_data]
        if missing_fields:
            return {'status': 'error', 'message': f"Missing required fields: {missing_fields}"}

        # 检查collection是否存在
        if not self.has_collection(collection_name):
            return {'status': 'error', 'message': f"Collection '{collection_name}' does not exist."}

        try:
            # *** 新增：处理 tags 字段，确保它是 List[str] ***
            try:
                record_data['tags'] = flatten_nested_structure(record_data['tags'])
                # 可选：进一步过滤掉非字符串或空的标签
                record_data['tags'] = [tag for tag in record_data['tags'] if isinstance(tag, str) and tag.strip()]
            except Exception as e:
                 return {'status': 'error', 'message': f"Failed to flatten tags: {str(e)}"}


            # 处理embedding字段-
            if not record_data.get("embedding", None) and auto_generate_embedding:
                try:
                    content = record_data['content']
                    # 假设有get_embedding方法，这里直接调用全局的模拟函数
                    record_data['embedding'] = get_embedding(content)
                except Exception as e:
                    return {'status': 'error', 'message': f"Failed to generate embedding: {str(e)}"}
            print(record_data)
            # 获取集合对象
            # 假设 Collection 类可以实例化并用于插入
            collection_orm = Collection(collection_name)
            collection_orm.load()

            # 执行插入
            insert_result = collection_orm.insert([record_data])
            inserted_count = getattr(insert_result, 'insert_count', 0)
            primary_keys = getattr(insert_result, 'primary_keys', [])

            # 可选刷新
            try:
                collection_orm.flush()
            except Exception as flush_e:
                print(f"Warning: Flush failed - {flush_e}")

            if inserted_count > 0:
                return {
                    'status': 'success',
                    'message': "Record inserted successfully",
                    'id': primary_keys[0] if primary_keys else None,
                    'embedding_generated': 'embedding' not in record_data and auto_generate_embedding
                }
            else:
                return {'status': 'error', 'message': "No records were inserted"}

        except Exception as e:
            return {'status': 'error', 'message': f"Insert operation failed: {str(e)}"}


     def insert_records(self, collection_name: str, records: List[Dict], auto_generate_embedding: bool = True, batch_size: int = 100) -> Dict[str, Any]:
          """
          批量插入记录，特性同insert_record

          参数:
               collection_name: 集合名称
               records: 记录列表
               auto_generate_embedding: 是否自动生成缺失的embedding
               batch_size: 分批处理的大小

          返回:
               操作结果字典
          """
          if not collection_name:
               return {'status': 'error', 'message': "Collection name cannot be empty."}
          if not records or not isinstance(records, list):
               return {'status': 'error', 'message': "No valid records provided."}

          if not self.has_collection(collection_name):
               return {'status': 'error', 'message': f"Collection '{collection_name}' does not exist."}

          # 预处理记录
          processed_records = []
          skipped_records = 0
          embedding_generated_count = 0

          for record in records:
               # 检查必需字段
               if not isinstance(record, dict) or 'content' not in record or 'tags' not in record:
                    skipped_records += 1
                    continue

               # *** 新增：处理 tags 字段，确保它是 List[str] ***
               try:
                    record['tags'] = flatten_nested_structure(record['tags'])
                    # 可选：进一步过滤掉非字符串或空的标签
                    record['tags'] = [tag for tag in record['tags'] if isinstance(tag, str) and tag.strip()]
               except Exception as e:
                    print(f"Failed to flatten tags for a record: {str(e)}")
                    skipped_records += 1
                    continue # 跳过此记录


               # 处理embedding
               if not record.get('embedding') and auto_generate_embedding:
                    try:
                         # 假设有get_embedding方法，这里直接调用全局的模拟函数
                         record['embedding'] = get_embedding(record['content'])
                         embedding_generated_count += 1
                    except Exception as e:
                         print(f"Failed to generate embedding for record: {str(e)}")
                         skipped_records += 1
                         continue

               processed_records.append(record)

          if not processed_records:
               return {
                    'status': 'error',
                    'message': "No valid records after processing",
                    'skipped_count': skipped_records
               }

          # 分批插入
          try:
               # 假设 Collection 类可以实例化并用于插入
               collection_orm = Collection(collection_name)
               collection_orm.load()

               total_inserted = 0
               for i in range(0, len(processed_records), batch_size):
                    batch = processed_records[i:i + batch_size]
                    insert_result = collection_orm.insert(batch)
                    total_inserted += getattr(insert_result, 'insert_count', 0)

               collection_orm.flush()

               return {
                    'status': 'success',
                    'inserted_count': total_inserted,
                    'skipped_count': skipped_records,
                    'embedding_generated_count': embedding_generated_count
               }

          except Exception as e:
               return {'status': 'error', 'message': f"Batch insert failed: {str(e)}"}

     def init_collection_structure(
          self,
          collection_name: str,
          drop_existing: bool = False,
          custom_schema: Optional[CollectionSchema] = None,
          index_params: Optional[Dict] = None,
          vector_field_name: str = "embedding"
          ) -> Dict[str, Any]:
               """
               高级功能：一步到位初始化 Collection 结构，包括创建 Collection 和创建索引。

               参数:
                    collection_name (str): 要初始化的 Collection 名称
                    drop_existing (bool): 是否删除已存在的同名 Collection (默认为 False)
                    custom_schema (Optional[CollectionSchema]): 自定义的 Collection schema (默认为 None，使用默认 schema)
                    index_params (Optional[Dict]): 自定义索引参数 (默认为 None，使用默认参数)
                    vector_field_name (str): 要创建索引的向量字段名 (默认为 "embedding")

               返回:
                    Dict[str, Any]: 包含操作状态和详细信息的字典
               """
               if not collection_name:
                    return {'status': 'error', 'message': "Collection name cannot be empty."}

               results = []
               final_status = "success"
               final_message = []

               # 1. 处理已存在的 Collection
               if drop_existing and self.has_collection(collection_name):
                    drop_result = self.drop_collection(collection_name)
                    results.append({'action': 'drop_collection', 'result': drop_result})
                    if drop_result['status'] == 'error':
                         return drop_result

               # 2. 创建 Collection
               if not self.has_collection(collection_name):
                    create_result = self.create_collection(
                         collection_name,
                         custom_schema=custom_schema
                    )
                    print(create_result)
                    results.append({'action': 'create_collection', 'result': create_result})
                    if create_result.get('status') == 'error':
                         return create_result
                    final_message.append(f"Collection '{collection_name}' created.")
               else:
                    final_message.append(f"Collection '{collection_name}' already exists.")

               # 3. 验证向量字段是否存在
               try:
                    collection_orm = Collection(collection_name)
                    schema = collection_orm.schema

                    # 获取所有向量字段
                    vector_fields = [
                         field.name for field in schema.fields
                         if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]
                    ]

                    if not vector_fields:
                         err_msg = f"No vector field found in collection '{collection_name}'"
                         return {'status': 'error', 'message': err_msg}

                    # 验证指定的向量字段是否存在
                    if vector_field_name not in vector_fields:
                         err_msg = (f"Specified vector field '{vector_field_name}' not found. "
                                   f"Available vector fields: {', '.join(vector_fields)}")
                         return {'status': 'error', 'message': err_msg}
               except Exception as e:
                    return {'status': 'error', 'message': f"Failed to validate schema: {str(e)}"}

               # 4. 创建索引
               try:
                    existing_indexes = collection_orm.indexes
                    if any(idx.field_name == vector_field_name for idx in existing_indexes):
                         msg = f"Index for field '{vector_field_name}' already exists."
                         final_message.append(msg)
                    else:
                         # 设置默认索引参数（如果未提供）
                         if index_params is None:
                              index_params = {
                              "index_type": "AUTOINDEX",
                              "metric_type": "COSINE",
                              "params": {}
                              }

                         index_result = self.create_collection_index(
                              collection_name,
                              field_name=vector_field_name,
                              index_params=index_params
                         )
                         results.append({'action': 'create_index', 'result': index_result})

                         if index_result['status'] == 'error':
                              final_status = 'warning'
                              final_message.append(f"Index creation failed: {index_result['message']}")
                         else:
                              final_message.append(f"Index for '{vector_field_name}' created successfully.")
               except Exception as e:
                    final_status = 'warning'
                    final_message.append(f"Error during index operation: {str(e)}")

               return {
                    'status': final_status,
                    'message': " ".join(final_message),
                    'details': results,
                    'vector_field': vector_field_name
               }


     # Example: drop_collection (can use MilvusClient)
     def drop_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        删除指定的 collection。
        使用 MilvusClient 进行删除。
        """
        if not collection_name or not self.client:
             return {'status': 'error', 'message': "Collection name cannot be empty or MilvusClient not initialized."}
        try:
             if self.client.has_collection(collection_name): # Use MilvusClient check
                  self.client.drop_collection(collection_name)
                  message = f"Collection '{collection_name}' dropped successfully."
                  print(message)
                  return {'status': 'success', 'message': message}
             else:
                  message = f"Collection '{collection_name}' does not exist, cannot drop."
                  print(message)
                  return {'status': 'info', 'message': message}
        except Exception as e:
             error_msg = f"Error dropping collection '{collection_name}' using MilvusClient: {e}"
             print(error_msg)
             traceback.print_exc()
             return {'status': 'error', 'message': error_msg}

     # Example: list_collections (can use MilvusClient)
     def list_collections(self) -> Dict[str, Any]:
          """
          列出所有的 collection。
          使用 MilvusClient 进行列出。
          """
          if not self.client:
               return {'status': 'error', 'message': "MilvusClient not initialized."}
          try:
               collections = self.client.list_collections()
               return {'status': 'success', 'collections': collections}
          except Exception as e:
               error_msg = f"Error listing collections using MilvusClient: {e}"
               print(error_msg)
               traceback.print_exc()
               return {'status': 'error', 'message': error_msg}

     # Example: create_collection_index (can use ORM Collection)
     def create_collection_index(self, collection_name: str, field_name: str = "embedding", index_params: Optional[Dict] = None) -> Dict[str, Any]:
          """
          为指定 Collection 的指定字段创建索引。
          依赖于 _connect_milvus_orm 中建立的 ORM 连接。
          使用 ORM Collection 的 create_index 方法。
          """
          if not collection_name:
               return {'status': 'error', 'message': "Collection name cannot be empty."}
          # Check collection exists (can use self.has_collection which uses MilvusClient or ORM utility)
          if not self.has_collection(collection_name):
               return {'status': 'error', 'message': f"Collection '{collection_name}' does not exist. Cannot create index."}

          # Ensure ORM connection is available for index creation
          if not Connections().has_connection(alias="default"):
               return {'status': 'error', 'message': "Milvus ORM connection not established. Cannot create index."}

          try:
               # Use Pymilvus ORM Collection object for index operation
               collection_orm = Collection(collection_name)

               # Check if index already exists using ORM
               existing_indexes = collection_orm.indexes
               if any(idx.field_name == field_name for idx in existing_indexes):
                    message = f"Index on field '{field_name}' for collection '{collection_name}' already exists."
                    print(message)
                    return {'status': 'info', 'message': message}

               if index_params is None:
                    index_params = {
                         "index_type": "AUTOINDEX",
                         "metric_type": "COSINE", # Or L2, IP，根据您的 embedding 模型选择
                         "params": {}
                    }

               print(f"Creating index for field '{field_name}' in collection '{collection_name}' using ORM...")
               collection_orm.create_index(field_name=field_name, index_params=index_params)
               message = f"Index created successfully for field '{field_name}' in collection '{collection_name}' using ORM."
               print(message)
               return {'status': 'success', 'message': message}
          except Exception as e:
               error_msg = f"Error creating index for collection '{collection_name}' using ORM: {e}"
               print(error_msg)
               traceback.print_exc()
               return {'status': 'error', 'message': error_msg}

     # Example: load_collection_to_memory (can use ORM Collection)
     def load_collection_to_memory(self, collection_name: str) -> Dict[str, Any]:
          """
          将指定 Collection 加载到内存以备搜索。
          依赖于 _connect_milvus_orm 中建立的 ORM 连接。
          使用 ORM Collection 的 load 方法。
          """
          if not collection_name:
               return {'status': 'error', 'message': "Collection name cannot be empty."}
          # Check collection exists
          if not self.has_collection(collection_name):
               return {'status': 'error', 'message': f"Collection '{collection_name}' does not exist. Cannot load to memory."}

          # Ensure ORM connection is available for loading
          if not Connections().has_connection(alias="default"):
               return {'status': 'error', 'message': "Milvus ORM connection not established. Cannot load collection."}

          try:
               # Use Pymilvus ORM Collection object for loading
               collection_orm = Collection(collection_name)
               print(f"Loading collection '{collection_name}' to memory using ORM...")
               collection_orm.load()
               message = f"Collection '{collection_name}' loaded into memory successfully using ORM."
               print(message)
               return {'status': 'success', 'message': message}
          except Exception as e:
               error_msg = f"Error loading collection '{collection_name}' to memory using ORM: {e}"
               print(error_msg)
               traceback.print_exc()
               return {'status': 'error', 'message': error_msg}