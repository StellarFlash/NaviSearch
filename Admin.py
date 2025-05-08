"""
Core.py

核心模块，负责调度SemanticTagger，SearchOperator等组件。

重构记录：
- 2025-04-29: 初始创建，实现基本的搜索功能。
- 2025-05-01: 添加了标签过滤功能，支持动态添加和移除标签。
- 2025-05-02: 优化了搜索逻辑，支持更复杂的查询。
- 2025-05-08: 进行了无状态改造，将状态管理移交给外部，同时改变了接口命名。
- 2025-05-07 (由AI助手重构): 将NaviSearchCore分离为NaviSearchAdmin与NaviSearchVisitor。
                 NaviSearchAdmin负责维护collection和加载数据。
                 NaviSearchVisitor负责搜索功能，并提供统一的search接口。
创建日期：2025-04-29
"""
import os
import traceback
import json
from typing import List, Dict, Optional, Any
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, Collection, DataType, Connections
from Search.SearchEngine import SearchEngine # 假设 SearchEngine 存在且路径正确
from Tagger.SemanticTagger import SemanticTagger # 假设 SemanticTagger 存在且路径正确
from utils import get_embedding, get_response, get_filter, flatten_nested_structure # 假设 utils 中的函数存在且路径正确

TAGGED_RECORDS_PATH = "Data/Chunks/5G行业应用安全评估规范+证明材料.jsonl"
RAW_CORPUS_PATH = "Data/Corpus/5G行业应用安全评估测试指引.jsonl"
DEFAULT_COLLECTION_NAME = "navi_search_collection"
DEFAULT_DIM = 1024
DEFAULT_MILVUS_TOKEN = "root:Milvus"
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"

class NaviSearchAdmin:
    """
    负责Milvus Collection的管理和数据加载。
    """
    def __init__(self,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 dim: int = DEFAULT_DIM,
                 milvus_token: str = DEFAULT_MILVUS_TOKEN,
                 milvus_host: str = DEFAULT_MILVUS_HOST,
                 milvus_port: str = DEFAULT_MILVUS_PORT,
                 tags_design_path: Optional[str] = None,
                 corpus_path: str = RAW_CORPUS_PATH):

        self.collection_name = collection_name
        self.dim = dim
        self.milvus_token = milvus_token
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.corpus_path = corpus_path

        self.client = MilvusClient(token=self.milvus_token, host=self.milvus_host, port=self.milvus_port)
        self.tagger = SemanticTagger(tags_design_path) if tags_design_path else None # Tagger只在Admin中用于数据预处理

        # 定义字段和Schema，这些是Collection的基础结构
        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="tags", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        self.schema = CollectionSchema(self.fields, description=f"Schema for {self.collection_name}")
        self._connect_milvus_orm() # 为了 Collection 对象的操作

    def _connect_milvus_orm(self):
        """连接Pymilvus ORM接口，用于索引创建等操作。"""
        try:
            Connections().connect(alias="default", host=self.milvus_host, port=self.milvus_port, token=self.milvus_token)
            print(f"Successfully connected to Milvus ORM interface at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            print(f"Failed to connect to Milvus ORM interface: {e}")
            # 根据实际需求，这里可能需要抛出异常或采取其他错误处理措施


    def init_and_load_collection(self, drop_existing: bool = False, use_cache: bool = True) -> Dict[str, Any]:
        """
        初始化Collection（如果不存在或指定删除），加载数据，并创建索引。
        """
        init_status = self.init_collection_structure(drop_existing=drop_existing)
        if init_status['status'] == 'error' and "已存在" not in init_status.get("message", "") : # 如果不是因为已存在而跳过创建，则返回错误
            return init_status

        load_status = self.load_corpus_to_collection(use_cache=use_cache)
        if load_status['status'] == 'error':
            return load_status

        index_status = self.create_collection_index()
        if index_status['status'] == 'error':
            return index_status

        load_to_memory_status = self.load_collection_to_memory()
        if load_to_memory_status['status'] == 'error':
            return load_to_memory_status

        return {
            'status': 'success',
            'message': f"Collection '{self.collection_name}' initialized, data loaded, indexed, and loaded to memory.",
            'details': {
                'init_collection': init_status.get('message'),
                'load_corpus': load_status.get('message'),
                'create_index': index_status.get('message'),
                'load_to_memory': load_to_memory_status.get('message')
            }
        }

    def init_collection_structure(self, drop_existing: bool = False) -> Dict[str, Any]:
        """初始化 Milvus collection 结构 (不包含数据加载和索引)"""
        try:
            if drop_existing and self.has_collection(self.collection_name):
                self.drop_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' dropped as per request.")

            if not self.has_collection(self.collection_name):
                self.client.create_collection(self.collection_name, schema=self.schema)
                message = f"Collection '{self.collection_name}' created successfully."
                print(message)
                return {'status': 'success', 'message': message}
            else:
                message = f"Collection '{self.collection_name}' already exists. Skipping creation."
                print(message)
                return {'status': 'info', 'message': message}
        except Exception as e:
            error_msg = f"Error during Milvus collection structure initialization for '{self.collection_name}': {e}"
            print(error_msg)
            return {'status': 'error', 'message': error_msg}

    def load_corpus_to_collection(self, use_cache: bool = True) -> Dict[str, Any]:
        """加载语料库到当前Collection"""
        if not self.corpus_path:
            return {'status': 'warning', 'message': "Corpus path not specified."}
        if not self.tagger and not use_cache: # 如果不用缓存且没有tagger，则无法处理原始数据
             return {'status': 'error', 'message': "SemanticTagger not initialized, cannot process raw corpus without cache."}

        try:
            corpus_data_to_insert = []
            message_prefix = ""
            if use_cache and os.path.exists(TAGGED_RECORDS_PATH):
                with open(TAGGED_RECORDS_PATH, 'r', encoding='utf-8') as f:
                    raw_data_from_file = [json.loads(line) for line in f]
                message_prefix = f"Using cached tagged records from {TAGGED_RECORDS_PATH}. "

                if raw_data_from_file and "page_content" in raw_data_from_file[0]: # 兼容旧的格式
                    for record in raw_data_from_file:
                        normalized_record_data = {
                            'content': record.get("page_content", ""),
                            'tags': flatten_nested_structure(record.get("metadata", {})),
                            'embedding': record.get("embedding") # 假设 embedding 已在缓存文件中
                        }
                        if not normalized_record_data['embedding']: # 如果缓存中没有 embedding
                             normalized_record_data['embedding'] = get_embedding(normalized_record_data['content'])
                        corpus_data_to_insert.append(normalized_record_data)
                else: # 假设是新格式或已经处理好的格式
                    for record in raw_data_from_file:
                        content = record.get("content", "")
                        embedding = record.get("embedding")
                        if not embedding and content: # 如果没有 embedding 则生成
                            embedding = get_embedding(content)
                        corpus_data_to_insert.append({
                            "content": content,
                            "tags": record.get("tags", []),
                            "embedding": embedding
                        })
            elif os.path.exists(self.corpus_path):
                message_prefix = f"Processing raw corpus from {self.corpus_path}. "
                with open(self.corpus_path, 'r', encoding='utf-8') as f:
                    raw_corpus_data = [json.loads(line) for line in f]

                if not self.tagger: # Should have been caught earlier but as a safeguard
                    return {'status': 'error', 'message': "SemanticTagger not initialized, cannot process raw corpus."}

                print(f"Tagging {len(raw_corpus_data)} records from raw corpus...")
                # 假设 tagger.RetrievalTimeTagging 返回 [{"content": "...", "tags": [...]}, ...]
                # 并且我们需要在这里为它们生成 embeddings
                # 注意：原代码在 RetrievalTimeTagging 后又打开 TAGGED_RECORDS_PATH 写入并生成embedding，
                # 这里简化为直接处理并插入。如果需要持久化带embedding的tagged_corpus，需要额外步骤。
                processed_records = []
                for record_content_dict in raw_corpus_data: # 假设原始数据是 {"content": "text"} 或类似结构
                    content = record_content_dict.get("content", "") # 或者根据实际格式调整
                    if not content: continue

                    # 模拟原 RetrievalTimeTagging 的输出结构，它可能只返回 content 和 tags
                    # 这里假设 tagger 返回的是带有 tags 的完整记录列表
                    # 或者，如果tagger只处理单个文本，需要迭代
                    tagged_record_list = self.tagger.RetrievalTimeTagging([{"content": content}]) # 假设输入是列表
                    if tagged_record_list:
                        for tagged_item in tagged_record_list: # RetrievalTimeTagging可能返回多个拆分后的项
                            tagged_item["embedding"] = get_embedding(tagged_item["content"])
                            processed_records.append(tagged_item)
                corpus_data_to_insert = processed_records

                # (可选) 持久化到 TAGGED_RECORDS_PATH
                try:
                    with open(TAGGED_RECORDS_PATH, 'w', encoding='utf-8') as f_out:
                        for record in corpus_data_to_insert:
                            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    print(f"Tagged records with embeddings saved to {TAGGED_RECORDS_PATH}")
                except Exception as e_save:
                    print(f"Warning: Could not save tagged records to cache: {e_save}")

            else:
                return {'status': 'warning', 'message': f"Neither cached file {TAGGED_RECORDS_PATH} nor raw corpus {self.corpus_path} found."}

            if not corpus_data_to_insert:
                return {'status': 'info', 'message': message_prefix + "No data to insert."}

            insert_message = self.insert_records(corpus_data_to_insert)
            if insert_message['status'] == 'success':
                # MilvusClient 的 insert 返回结果包含 insert_count 和 ids
                num_inserted = insert_message['message'].get('insert_count', len(corpus_data_to_insert)) # 获取实际插入数量
                final_message = message_prefix + f"Successfully inserted {num_inserted} records into '{self.collection_name}'."
                print(final_message)
                # self.client.flush([self.collection_name]) # MilvusClient v2.4+ 已移除 flush
                Collection(self.collection_name).flush() # 使用 Collection 对象 flush
                print(f"Collection '{self.collection_name}' flushed.")
                return {'status': 'success', 'message': final_message, 'insert_details': insert_message['message']}
            else:
                return {'status': 'error', 'message': message_prefix + f"Failed to insert records: {insert_message['message']}"}

        except Exception as e:
            error_msg = f"Error loading corpus to collection '{self.collection_name}': {e}"
            traceback.print_exc()
            return {'status': 'error', 'message': error_msg}


    def create_collection_index(self, field_name: str = "embedding", index_params: Optional[Dict] = None) -> Dict[str, Any]:
        """为当前Collection的embedding字段创建索引"""
        if not self.has_collection(self.collection_name):
            return {'status': 'error', 'message': f"Collection '{self.collection_name}' does not exist. Cannot create index."}
        try:
            collection_orm = Collection(self.collection_name) # 使用Pymilvus ORM Collection
            # 检查是否已有索引
            if any(idx.field_name == field_name for idx in collection_orm.indexes):
                 message = f"Index on field '{field_name}' for collection '{self.collection_name}' already exists."
                 print(message)
                 return {'status': 'info', 'message': message}

            if index_params is None:
                index_params = {
                    "index_type": "AUTOINDEX", # 或者 "IVF_FLAT", "HNSW" 等
                    "metric_type": "L2",      # 对于 AUTOINDEX, metric_type 通常在创建 collection 时指定，或让 Milvus 自动选择
                    "params": {}              # 对于 AUTOINDEX, params 通常为空或根据需要配置
                }
                # 如果使用 IVF_FLAT:
                # index_params = {
                #     "index_type": "IVF_FLAT",
                #     "metric_type": "L2",
                #     "params": {"nlist": 128} # nlist 通常是 sqrt(N) 到 4*sqrt(N) 之间, N是总记录数
                # }

            print(f"Creating index for field '{field_name}' in collection '{self.collection_name}' with params: {index_params}...")
            collection_orm.create_index(field_name=field_name, index_params=index_params)
            message = f"Index created successfully for field '{field_name}' in collection '{self.collection_name}'."
            print(message)
            return {'status': 'success', 'message': message}
        except Exception as e:
            error_msg = f"Error creating index for collection '{self.collection_name}': {e}"
            traceback.print_exc()
            return {'status': 'error', 'message': error_msg}

    def load_collection_to_memory(self) -> Dict[str, Any]:
        """将当前Collection加载到内存以备搜索"""
        if not self.has_collection(self.collection_name):
            return {'status': 'error', 'message': f"Collection '{self.collection_name}' does not exist. Cannot load to memory."}
        try:
            collection_orm = Collection(self.collection_name)
            # 检查加载状态，避免重复加载或在未加载时尝试释放
            # Milvus 2.3+ Collection 对象有 `load_state` 属性，但 Pymilvus 的 Collection 对象可能没有直接的 load_state。
            # 通常，load() 操作是幂等的或有内部处理。
            print(f"Loading collection '{self.collection_name}' to memory...")
            collection_orm.load()
            message = f"Collection '{self.collection_name}' loaded into memory successfully."
            print(message)
            return {'status': 'success', 'message': message}
        except Exception as e:
            error_msg = f"Error loading collection '{self.collection_name}' to memory: {e}"
            traceback.print_exc()
            return {'status': 'error', 'message': error_msg}

    def create_collection(self, collection_name: str) -> Dict[str, Any]:
        """创建一个新的 collection (仅结构，不加载数据或索引)"""
        if not collection_name:
            return {'status': 'error', 'message': "Collection name cannot be empty."}
        try:
            if self.has_collection(collection_name):
                message = f"Collection '{collection_name}' already exists."
                print(message)
                return {'status': 'info', 'message': message}
            else:
                # 使用类级别的 schema 定义
                self.client.create_collection(collection_name, schema=self.schema)
                message = f"Collection '{collection_name}' created successfully."
                print(message)
                return {'status': 'success', 'message': message}
        except Exception as e:
            error_msg = f"Error creating collection '{collection_name}': {e}"
            print(error_msg)
            return {'status': 'error', 'message': error_msg}

    def drop_collection(self, collection_name: str) -> Dict[str, Any]:
        """删除指定的 collection"""
        if not collection_name:
            return {'status': 'error', 'message': "Collection name cannot be empty."}
        try:
            if self.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                message = f"Collection '{collection_name}' dropped successfully."
                print(message)
                return {'status': 'success', 'message': message}
            else:
                message = f"Collection '{collection_name}' does not exist, cannot drop."
                print(message)
                return {'status': 'info', 'message': message}
        except Exception as e:
            error_msg = f"Error dropping collection '{collection_name}': {e}"
            print(error_msg)
            return {'status': 'error', 'message': error_msg}

    def list_collections(self) -> Dict[str, Any]:
        """列出所有的 collection"""
        try:
            collections = self.client.list_collections()
            return {'status': 'success', 'collections': collections}
        except Exception as e:
            error_msg = f"Error listing collections: {e}"
            print(error_msg)
            return {'status': 'error', 'message': error_msg}

    def get_current_target_collection_name(self) -> str:
        """获取当前Admin实例配置的collection名称"""
        return self.collection_name

    def set_target_collection_name(self, collection_name: str) -> Dict[str, Any]:
        """
        设置Admin实例将要操作的collection名称。
        注意：这只改变实例的内部状态，实际操作时需确保该collection存在。
        """
        if not collection_name:
            return {'status': 'error', 'message': "New collection name cannot be empty."}
        if self.has_collection(collection_name):
            self.collection_name = collection_name
            # 更新 schema 描述以反映新的 collection 名称
            self.schema = CollectionSchema(self.fields, description=f"Schema for {self.collection_name}")
            message = f"Admin target collection changed to: '{collection_name}'."
            print(message)
            return {'status': 'success', 'message': message}
        else:
            message = f"Warning: Collection '{collection_name}' does not exist. Set as target, but operations might fail until created."
            self.collection_name = collection_name
            self.schema = CollectionSchema(self.fields, description=f"Schema for {self.collection_name}")
            print(message)
            return {'status': 'warning', 'message': message}


    def has_collection(self, collection_name: str) -> bool:
        """检查是否存在指定的 collection"""
        try:
            return self.client.has_collection(collection_name)
        except Exception as e:
            print(f"Error checking if collection '{collection_name}' exists: {e}")
            return False

    def insert_record(self, record_data: Dict, target_collection: Optional[str] = None) -> Dict[str, Any]:
        """插入一条记录到指定的或当前的collection"""
        collection_to_insert = target_collection or self.collection_name
        if not self.has_collection(collection_to_insert):
            return {'status': 'error', 'message': f"Collection '{collection_to_insert}' does not exist."}
        try:
            content = record_data.get("content", "Default test content.")
            embedding = record_data.get("embedding", get_embedding(content)) # 自动生成 embedding (如果未提供)
            tags = record_data.get("tags", [])

            data_to_insert = {
                "content": content,
                "embedding": embedding,
                "tags": tags
            }
            # upsert 返回的是 MutationResult 对象
            result = self.client.upsert(
                collection_name=collection_to_insert,
                data=data_to_insert, # upsert 单条记录时，data应该是dict
            )
            # primary_keys = result.primary_keys # MilvusClient 2.4+
            # MilvusClient v2.2.x, result is a dict {'ids': [...]}
            inserted_id = result.get("ids", [None])[0] if isinstance(result, dict) else result.get("primary_keys")[0]

            return {
                'status': 'success',
                'message': f"Record inserted/upserted successfully into '{collection_to_insert}'.",
                'id': inserted_id
            }
        except Exception as e:
            error_msg = f"Error inserting record into '{collection_to_insert}': {e}"
            traceback.print_exc()
            return {'status': 'error', 'message': error_msg}

    def insert_records(self, records: List[Dict], target_collection: Optional[str] = None) -> Dict[str, Any]:
        """批量插入记录到指定的或当前的collection"""
        collection_to_insert = target_collection or self.collection_name
        if not self.has_collection(collection_to_insert):
            return {'status': 'error', 'message': f"Collection '{collection_to_insert}' does not exist."}
        if not records:
            return {'status': 'info', 'message': "No records provided to insert."}
        try:
            data_to_insert = []
            for record in records:
                content = record.get("content", "")
                embedding = record.get("embedding")
                if embedding is None or not embedding: # 检查 embedding 是否有效
                    if content: # 只有在 content 存在时才尝试生成 embedding
                        embedding = get_embedding(content)
                    else: # 如果 content 也为空，则这条记录无法有效插入向量库
                        print(f"Warning: Skipping record due to empty content and no embedding: {record}")
                        continue # 跳过这条记录
                data_to_insert.append({
                    "content": content,
                    "tags": record.get("tags", []),
                    "embedding": embedding,
                })

            if not data_to_insert: # 如果处理后没有可插入的数据
                 return {'status': 'info', 'message': "No valid records to insert after processing."}

            # client.insert 返回的是 MutationResult 对象
            insert_result = self.client.insert(
                collection_name=collection_to_insert,
                data=data_to_insert,
            )
            # insert_result 包含 primary_keys 和 insert_count
            # MilvusClient v2.2.x, insert_result is a dict {'ids': [...], 'insert_count': X}
            # MilvusClient v2.4.x, insert_result is MutationResult with .primary_keys and .insert_count
            if isinstance(insert_result, dict): #兼容旧版
                count = insert_result.get('insert_count', len(insert_result.get('ids',[])))
                pks = insert_result.get('ids',[])
            else: #新版 Pymilvus
                count = insert_result.get("insert_count", "Unknown")
                pks = insert_result.get("primary_keys", [])

            return {
                'status': 'success',
                "message": { # 返回更详细的结果
                    "insert_count": count,
                    "primary_keys": pks,
                    "details": f"Successfully inserted {count} records into '{collection_to_insert}'."
                }
            }
        except Exception as e:
            error_msg = f"Error bulk inserting records into '{collection_to_insert}': {e}"
            traceback.print_exc()
            return {'status': 'error', 'message': error_msg}