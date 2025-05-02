"""

NaviSearchCore.py

核心模块，负责调度SemanticTagger，SearchOperator等组件。

创建日期：2025-04-29
"""
import traceback
import json
from typing import List, Dict, Optional
from pymilvus import MilvusClient,FieldSchema,CollectionSchema,Collection,DataType,Connections
from Search.SearchEngine import SearchEngine
from Tagger.SemanticTagger import SemanticTagger
from utils import get_embedding, get_response, get_filter



class NaviSearchCore:
    def __init__(self, tags_design_path=None, corpus_path="Data/Corpus/5G行业应用安全评估测试指引.jsonl"):

        self.corpus_path = corpus_path
        self.client = MilvusClient(
            token = "root:Milvus"
        )
        self.collection_name = "navi_search_collection"
        self.dim = 1024

        self.tagger = SemanticTagger(tags_design_path)
        self.active_tags = []  # 核心维护当前激活的标签
        self.init_collection()
        self.search_engine = SearchEngine(
            client = self.client, collection_name = self.collection_name
        )


        # self.create_collection(self.collection_name)
        # self.load_corpus()


    def init_collection(self, drop_existing=True):
        """初始化 Milvus collection"""
        try:
            self.drop_collection(self.collection_name) if drop_existing else None
        except Exception as e:
            print(f"删除 Milvus collection 时发生错误: {e}，试图删除一个不存在的collection。")
        try:
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 字符串内容
                FieldSchema(name="tags", dtype=DataType.JSON),  # JSON 类型 tags
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)  # 1024维 embedding
            ]

            # 创建 Schema
            self.schema = CollectionSchema(fields, description="Schema for document embeddings")

            # 创建Collection
            self.client.create_collection(self.collection_name, schema=self.schema)
            connection = Connections()
            connection.connect(alias="default", host="localhost", port="19530",token="root:Milvus")
            self.collection = Collection(self.collection_name)  # 初始化 collection 实例
              # 连接到 Milvus 服务，确保连接别名与 MilvusClient 中的一致
        except Exception as e:
            print(f"创建 Milvus collection 时发生错误: {e}")

        self.load_corpus()

         # ✅ 数据插入完成后，创建索引
        collection = Collection(name=self.collection_name)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 100}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("索引创建完成")

        # 加载到内存
        collection.load()
        print("集合已加载到内存，准备检索")

    def load_corpus(self, use_cache = True):
        """加载语料库"""
        if self.corpus_path == "":
            return{
                "status": 'warning',
                "message": "未指定语料库路径"
            }
        try:
            if use_cache:
                with open('Data/Records/5G行业应用安全评估测试指引_tagged.jsonl', 'r', encoding='utf-8') as f:
                    corpus_data = [json.loads(line) for line in f]
                    self.insert_records(corpus_data)  # 调用 insert_records 方法插入语料库数据
                    return {
                        "status": 'success',
                        "message": f"已加载缓存语料库 {self.corpus_path}中的{len(corpus_data)}条记录。"
                    }
            # 检查是否有record文件
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = [json.loads(line) for line in f]
                tagged_corpus_data = self.tagger.RetrievalTimeTagging(corpus_data)  # 调用 RetrievalTimeTagging 方法生成标签
                # 持久化到本地文件
                with open('Data/Records/5G行业应用安全评估测试指引_tagged.jsonl', 'w', encoding='utf-8') as f:
                    for index, record in enumerate(tagged_corpus_data):
                        record["embedding"] = get_embedding(record["content"])  # 生成 embedding，假设 get_embedding 是一个函数来生成 embedding
                        tagged_corpus_data[index]["embedding"] = record["embedding"]
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

                self.insert_records(corpus_data)  # 调用 insert_records 方法插入语料库数据
                return {
                    "status": 'success',
                    "message": f"已加载语料库 {self.corpus_path}中的{len(corpus_data)}条记录。"
                }
        except Exception as e:
            return{
                "status": 'error',
                "message": f"加载语料库失败: {e}"
            }
    # collection相关的方法
    def create_collection(self, collection_name:str = ""):
        """创建一个新的 collection"""
        if collection_name == "":
            collection_name = self.collection_name
        else:
            self.collection_name = collection_name
        try:
            if self.has_collection(collection_name):
                print(f"collection {collection_name} 已存在，无需创建。")
                return {
                    'status': 'success',
                    "message": f"collection {collection_name} 已存在，无需创建。"
                }
            else:
                self.client.create_collection(collection_name, schema = self.schema)
                print(f"已创建 collection: {collection_name}")
                return {
                    'status':'success',
                    "message": f"已创建 collection: {collection_name}"
                }
        except Exception as e:
            print(f"创建 collection {collection_name} 时发生错误: {e}")

    def drop_collection(self, collection_name:str = ""):
        """删除指定的 collection"""
        if collection_name == "":
            collection_name = self.collection_name
        else:
            self.collection_name = collection_name
        try:
            if self.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                print(f"已删除 collection: {collection_name}")
                return {
                  'status':'success',
                  'message': f"已删除 collection: {collection_name}"
                }
            else:
                print(f"collection {collection_name} 不存在，无法删除。")
                return {
                    'status': 'fail',
                    'message': f"collection {collection_name} 不存在，无法删除。"
                }
        except Exception as e:
            print(f"删除 collection {collection_name} 时发生错误: {e}")
            return {
                'status': 'error',
                'message': f"删除 collection {collection_name} 时发生错误: {str(e)}"
            }

    def list_collections(self):
        """列出所有的 collection"""
        try:
            collections = self.client.list_collections()
            return {
              'status':'success',
              'collections': collections
            }
        except Exception as e:
            error_msg = f"列出 collection 时发生错误: {str(e)}"
            print(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }
    def get_current_collection(self):
        """获取当前使用的 collection 名称"""
        return self.collection_name

    def use_collection(self, collection_name:str = ""):
        """切换到指定的 collection"""
        try:
            if not self.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} 不存在。")
            self.collection_name = collection_name  # 更新当前实例的 collection 名称
            print(f"已切换到 collection: {collection_name}")
            return {
                'status': 'success',
                'message': f"已切换到 collection: {collection_name}"
            }
        except Exception as e:
            print(f"切换 collection 失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def has_collection(self, collection_name):
        """检查是否存在指定的 collection"""
        try:
            return self.client.has_collection(collection_name)
        except Exception as e:
            print(f"检查 collection 失败: {e}")
            return False


    # tags 相关的方法
    def add_tags(self, tag_content_to_parse):
        if not tag_content_to_parse:
            return "未提供有效标签内容。"
        # 使用逗号分割，并去除空格
        tags = [t.strip() for t in tag_content_to_parse.split(',') if t.strip()]
        if not tags:
            return "未提供有效标签。"

        new_tags = [t for t in tags if t not in self.active_tags]
        if new_tags:
            self.active_tags.extend(new_tags)
            return f"新增标签: {new_tags}"
        else:
            return "提供的标签已存在或为空。"

    def clear_tags(self):
        self.active_tags.clear()
        return "已清空所有标签。"

    def get_active_tags(self):
        return self.active_tags.copy()

    # Record 相关的方法
    def insert_record(self, record_data: Dict = {"tags":None,"content":"这是一条测试数据。","embedding":[0.0]*1024}) -> Dict:
        """插入一条记录"""
        try:
            # 准备基础数据（确保tags存在）
            base_data = {
                "content": record_data.get("content", "这是一条测试数据。"),
                "embedding": record_data.get("embedding", get_embedding(record_data.get("content"))),
                "tags": record_data.get("tags", []),
            }
            result = self.client.upsert(
                collection_name=self.collection_name,
                data=base_data,
            )
            id = result.get("ids",[0])[0]
            return {
               'status':'success',
               'message': "记录插入成功",
               'id': id
            }
        except Exception as e:
            return {
               'status': 'error',
               'message': str(e)
            }

    def insert_records(self, records:List[Dict] = None)->Dict:
        """
        批量插入记录。

        Args:
            records:
                content: str, 记录内容
                tags: List[str], 记录标签
                embedding: List[float], 嵌入向量

        Returns:
            加载的EvaluationSpec对象
        """
        try:
            data = []
            for record in records:
                content = record.get("content","")
                tags = record.get("tags",[])
                embedding = record.get("embedding")
                if embedding is None or embedding == []:
                    embedding = get_embedding(content)
                data.append({
                    "content": content,
                    "tags": tags,
                    "embedding": embedding,
                })
            # print(data[0])
            # exit()
            insert_result = self.client.insert(
                collection_name=self.collection_name,
                data=data,
            )
            return {
                'status':'success',
                "message": insert_result
            }
        except Exception as e:
            return {
               'status': 'error',
               'message': str(e)
            }

    # 搜索相关的方法
    def perform_search(self, query_text):
        """进行搜索并返回结果"""
        active_filter_tags = self.active_tags if self.active_tags else None
        try:
            retrieval_records = self.search_engine.retrieval(
                query_text=query_text,
                top_k=20
            )
            # print(f"初次召回 {len(retrieval_records)} 个结果。")
            ranked_records, ranked_tags = self.search_engine.rerank(
                filter_tags=active_filter_tags,
                retrieval_records = retrieval_records,
                mode = "ranking",
                top_k = 5
            )
            print(f"剩余 {len(ranked_records)} 个结果。")

            return {
                'status': 'success',
                'ranked_records': ranked_records[:5],
                'ranked_tags': ranked_tags[:10]
            }
        except Exception as e:
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }

    def perform_filter_search(self, query_text:str = "", retrieval_size:str = 20, stop_size = 3, max_iteration = 10):
        """
        使用LLM代替用户选择过滤标签。
        """

        retrieval_records = self.search_engine.retrieval(query_text = query_text, top_k = retrieval_size)

        remaining_records = retrieval_records.copy()
        current_filter = []
        remaining_records, recomened_tags = self.search_engine.rerank(
            filter_tags = current_filter,
            retrieval_records = remaining_records,
            mode = "filtering",
            top_k = stop_size
        )
        print(f"初次召回 {len(retrieval_records)} 个结果。")
        current_iteration = 0
        for i in range(max_iteration):
            # 生成过滤标签
            current_filter = get_filter(
                query_text = query_text,
                current_filter = current_filter,
                recomanded_filter = recomened_tags,
                current_iteration = current_iteration,
                current_size = len(remaining_records),
                max_iteration = max_iteration,
                stop_size = stop_size
            )
            # 过滤记录
            remaining_records, recomened_tags = self.search_engine.rerank(
                filter_tags = current_filter,
                retrieval_records = remaining_records,
                mode = "filtering",
                top_k = stop_size
            )
            print(f"第{i+1}次迭代。")
            print(f"过滤标签：{current_filter}")
            print(f"剩余 {len(remaining_records)} 个结果。")
            if len(remaining_records) <= stop_size:
                import os
                # os.system('cls')
                print("*"*50)
                print(f"最终过滤标签：{current_filter}")
                print(f"{remaining_records[0]['content']}")
                return {
                   'status':'success',
                    'ranked_records': remaining_records,
                    'filter_tags': current_filter
                }
        return {
            'status':'fail',
            'ranked_records': remaining_records,
            'filter_tags': current_filter
        }


if __name__ == "__main__":
    core = NaviSearchCore(tags_design_path="Data/Tags/tags_design.json")
    query = """N4接口需要进行哪些安全评估？。
"""
    retrieval_response = core.perform_filter_search(query_text = query)
    retrieval_str = retrieval_response.get("ranked_records")
    print(get_response(prompt = f"根据检索结果回答用户问题。\n用户问题：{query}\n召回结果：{retrieval_response['ranked_records'][0]['content']}\n回答："))