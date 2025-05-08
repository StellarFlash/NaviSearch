"""
Visitor.py

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


DEFAULT_COLLECTION_NAME = "navi_search_collection"
DEFAULT_MILVUS_TOKEN = "root:Milvus"
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"

class NaviSearchVisitor:
    """
    负责搜索功能和用户会话相关的标签管理。
    """
    def __init__(self,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 milvus_token: str = DEFAULT_MILVUS_TOKEN,
                 milvus_host: str = DEFAULT_MILVUS_HOST,
                 milvus_port: str = DEFAULT_MILVUS_PORT):

        self.collection_name = collection_name
        self.active_tags = []
        # Visitor 也需要 Milvus client 来操作 SearchEngine
        self.client = MilvusClient(token=milvus_token, host=milvus_host, port=milvus_port)
        self.connection = Connections()
        # 初始化时不直接连接
        self._is_connected = False
        self.collection = None

    def connect_milvus(self) -> Dict[str, Any]:
        """连接Milvus数据库"""
        if self._is_connected:
            return {'status': 'success', 'message': "Already connected to Milvus."}

        try:
            self.connection.connect(alias = "default", host = DEFAULT_MILVUS_HOST, port = DEFAULT_MILVUS_PORT, token = DEFAULT_MILVUS_TOKEN)
            self.collection = Collection(self.collection_name)
            if not self.client.has_collection(self.collection_name):
                print(f"Warning: Collection '{self.collection_name}' does not exist.")
            self._is_connected = True
            return {'status': 'success', 'message': "Connected successfully."}
        except Exception as e:
            return {'status':'error', 'message': f"Failed to connect: {str(e)}"}

    def disconnect_milvus(self) -> Dict[str, Any]:
        """断开与Milvus的连接"""
        try:
            self.connection(alias = "default") # 断开连接
            return {'status':'success','message': "Disconnected from Milvus successfully."}
        except Exception as e:
            error_message = f"Failed to disconnect from Milvus: {str(e)}"
        return {'status':'error','message': error_message}

    def set_target_collection_name(self, collection_name: str) -> Dict[str, Any]:
        """切换Visitor实例搜索的目标collection名称"""
        if not collection_name:
            return {'status': 'error', 'message': "New collection name cannot be empty."}

        if not self.client.has_collection(collection_name):
            message = f"Warning: Collection '{collection_name}' does not exist. Search operations will likely fail until it's available and loaded."
            print(message)
            # 即使不存在，也允许切换，让后续操作失败来提示用户
            self.collection_name = collection_name
            self.search_engine = SearchEngine(client=self.client, collection_name=self.collection_name) # 重新初始化SearchEngine
            return {'status': 'warning', 'message': message}
        else:
            # 确保 Collection 已加载到内存 (Visitor 通常不负责加载，但检查是有益的)
            # Pymilvus Collection对象可以查询加载状态，但 MilvusClient 没有直接接口
            # 这里假设Collection已由Admin加载。
            # Collection(collection_name).load() # Visitor不应该执行load操作
            pass

        self.collection_name = collection_name
        # 需要重新初始化 SearchEngine 或更新其 collection_name
        self.search_engine = SearchEngine(client=self.client, collection_name=self.collection_name)
        message = f"Visitor target collection changed to: '{collection_name}'."
        print(message)
        # 清空当前会话的激活标签，因为它们可能与新的collection不相关
        self.clear_tags()
        print("Active tags cleared due to collection change.")
        return {'status': 'success', 'message': message}

    def get_current_target_collection_name(self) -> str:
        """获取当前Visitor实例配置的collection名称"""
        return self.collection_name

    def add_tags(self, tag_content_to_parse: str) -> Dict[str, Any]:
        """添加激活标签，用于后续搜索的过滤"""
        if not isinstance(tag_content_to_parse, str) or not tag_content_to_parse.strip():
            return {'status': 'warning', 'message': "No valid tag content provided."}

        tags = [t.strip() for t in tag_content_to_parse.split(',') if t.strip()]
        if not tags:
            return {'status': 'warning', 'message': "No valid tags extracted from content."}

        newly_added_tags = []
        existing_tags_provided = []
        for t in tags:
            if t not in self.active_tags:
                self.active_tags.append(t)
                newly_added_tags.append(t)
            else:
                existing_tags_provided.append(t)

        message_parts = []
        if newly_added_tags:
            message_parts.append(f"Added new tags: {newly_added_tags}.")
        if existing_tags_provided:
            message_parts.append(f"Tags already active: {existing_tags_provided}.")

        if not message_parts: # Should not happen if tags were valid
            return {'status': 'info', 'message': "No changes to active tags."}

        return {
            'status': 'success',
            'message': " ".join(message_parts),
            'active_tags': self.active_tags.copy()
        }

    def clear_tags(self) -> Dict[str, Any]:
        """清空所有激活标签"""
        self.active_tags.clear()
        return {'status': 'success', 'message': "All active tags cleared.", 'active_tags': []}

    def get_active_tags(self) -> List[str]:
        """获取当前激活的标签列表"""
        return self.active_tags.copy()

    def search(self,
               query_text: str,
               filter_tags: List[str] = None,
               mode: str = "standard", # "standard" or "agentic"
               retrieval_top_k: int = 20,
               rerank_strategy: Optional[str] = "ranking", # maps to rerank mode like "ranking" or "filtering"
               rerank_doc_top_k: Optional[int] = 5, # for standard_search
               rerank_tags_top_k: Optional[int] = 10,
               **kwargs: Any) -> Dict[str, Any]:
        """
        统一的搜索接口。

        Args:
            query_text (str): 用户查询。
            mode (str): "standard_search" (人工指定标签过滤) 或 "agentic_search" (LLM辅助标签过滤)。
            retrieval_top_k (int): 初步向量检索召回的数量。
            rerank_strategy (str, optional): 重排阶段的策略。
                                            对于 standard_search，通常是 "ranking"。
                                            对于 agentic_search，通常是 "filtering"。
            **kwargs:
                standard_search:
                    rerank_top_k (int): 重排后返回的记录数量 (默认为5)。
                agentic_search:
                    rerank_stop_size (int): agentic 搜索迭代停止的记录数量阈值 (默认为3)。
                    max_iterations (int): agentic 搜索的最大迭代次数 (默认为10)。
        """
        if not self.client.has_collection(self.collection_name):
             return {'status': 'error', 'message': f"Collection '{self.collection_name}' not found or not loaded. Cannot perform search."}
        if filter_tags:
            active_filter_tags = filter_tags
        else:
            active_filter_tags = self.get_active_tags() # standard_search 会使用这里激活的标签

        if mode == "standard":
            rerank_top_k = kwargs.get("rerank_top_k", 5)
            try:
                # 1. 初步检索
                retrieval_records = self.search_engine.retrieval(
                    query_text=query_text,
                    top_k=retrieval_top_k
                    # 注意：原版 SearchEngine.retrieval 可能不直接支持 filter_tags,
                    # 过滤通常在 rerank 阶段或 Milvus 查询时通过 expr 参数实现。
                    # 如果 SearchEngine.retrieval 支持 expr，可以在此传入基于 active_filter_tags 的表达式。
                    # 否则，过滤将在 rerank 阶段处理。
                )
                print(f"Initial retrieval returned {len(retrieval_records)} records for standard search.")
                if not retrieval_records:
                    return {
                        'status': 'success',
                        'message': 'No records found in initial retrieval.',
                        'ranked_records': [],
                        'ranked_tags': []
                    }

                # 2. 重排和过滤 (使用 active_tags)
                # rerank_strategy 对应 SearchEngine.rerank 的 mode 参数
                ranked_records, ranked_tags = self.search_engine.rerank(
                    filter_tags=active_filter_tags if active_filter_tags else None, # 明确传递None或列表
                    retrieval_records=retrieval_records,
                    mode=rerank_strategy or "ranking", # "ranking"
                    top_k=rerank_top_k # rerank后保留的数量
                )
                print(f"Reranking (strategy: {rerank_strategy or 'ranking'}) with active tags {active_filter_tags} resulted in {len(ranked_records)} records.")

                return {
                    'status': 'success',
                    'ranked_records': ranked_records, # rerank_top_k 已经在这里生效
                    'ranked_tags': ranked_tags[:10], # 假设推荐标签也取前10
                    'active_filter_tags_used': active_filter_tags
                }
            except Exception as e:
                traceback.print_exc()
                return {'status': 'error', 'message': f"Error in standard_search: {str(e)}"}

        elif mode == "agentic":
            rerank_stop_size = kwargs.get("rerank_stop_size", 3)
            max_iterations = kwargs.get("max_iterations", 10)
            try:
                # 1. 初步检索
                initial_retrieval_records = self.search_engine.retrieval(
                    query_text=query_text,
                    top_k=retrieval_top_k
                )
                print(f"Initial retrieval returned {len(initial_retrieval_records)} records for agentic search.")
                if not initial_retrieval_records:
                    return {
                        'status': 'success',
                        'message': 'No records found in initial retrieval for agentic search.',
                        'ranked_records': [],
                        'filter_tags': []
                    }

                remaining_records = initial_retrieval_records.copy()
                current_llm_filter_tags: List[str] = [] # LLM选择的标签

                # 第一次 rerank/filtering 获取初始推荐标签
                # rerank_strategy 对应 SearchEngine.rerank 的 mode 参数
                # 对于 agentic search 的迭代，SearchEngine.rerank 的 mode 通常是 "filtering"
                # top_k 在这里是 rerank_stop_size，因为 rerank 方法的 top_k 是输出数量
                remaining_records, recommended_tags = self.search_engine.rerank(
                    filter_tags=current_llm_filter_tags, # 初始为空
                    retrieval_records=remaining_records,
                    mode=rerank_strategy or "filtering", # "filtering"
                    top_k=len(remaining_records) # 第一次不过滤数量，只获取推荐标签和排序
                )
                print(f"After initial rerank (mode: {rerank_strategy or 'filtering'}), {len(remaining_records)} records remain. Recommended tags: {recommended_tags[:5]}")


                for i in range(max_iterations):
                    # 2. LLM 生成过滤标签 (get_filter)
                    current_llm_filter_tags = get_filter( # 假设 get_filter 返回 List[str]
                        query_text=query_text,
                        current_filter=current_llm_filter_tags,
                        recomanded_filter=recommended_tags, # 注意：原参数名 recomAnded_filter，已修正
                        current_iteration=i,
                        current_size=len(remaining_records),
                        max_iteration=max_iterations, # 原参数名 max_iteration
                        stop_size=rerank_stop_size    # 原参数名 stop_size
                    )
                    if not isinstance(current_llm_filter_tags, list): # 确保是列表
                        print(f"Warning: get_filter did not return a list, got: {current_llm_filter_tags}. Assuming no filter.")
                        current_llm_filter_tags = []


                    # 3. 使用LLM选择的标签进行过滤 (rerank)
                    # rerank_strategy 应该持续是 "filtering"
                    # top_k 控制 rerank 后的输出数量，这里可以设为 len(remaining_records)
                    # 因为我们是基于标签过滤，而不是按数量截断，除非到了最后一步。
                    # 或者，让 rerank 函数的 top_k 参数用于最终截断。
                    # SearchEngine.rerank 的 top_k 参数会影响返回的记录数量。
                    # 如果我们想在过滤后保留所有符合条件的，然后再看是否小于 stop_size，
                    # 那么 rerank 的 top_k 应设得较大，或等于当前记录数。
                    # 然后在循环外或判断 len(remaining_records) <= rerank_stop_size 时再截断。
                    # 这里的 rerank_stop_size 是 agentic search 的停止条件，不是 rerank 方法的 top_k。
                    # rerank 方法的 top_k 应该是保留多少条结果。
                    # 在这里，我们先按标签过滤，然后检查数量。
                    # rerank的mode参数是 "filtering" 时，其 top_k 应该是指过滤后保留的记录数。
                    # 但原代码中 rerank 的 top_k 是 stop_size，这可能意味着 rerank 内部也做了数量控制。
                    # 为了匹配原逻辑，我们将 rerank 的 top_k 参数设置为当前 remaining_records 的长度，
                    # 以确保 rerank 只按标签过滤，不减少数量，数量判断在循环中进行。

                    if not current_llm_filter_tags: # 如果LLM没有给出任何标签，可能意味着无法进一步过滤或已满足
                        print(f"Iteration {i+1}: LLM provided no new filter tags. Agentic search may conclude.")
                        # 可以选择在此处中断，或者让循环自然结束/满足stop_size
                        # For now, let it continue, it might hit stop_size or max_iterations
                        # Or, if no tags and size > stop_size, then it's stuck.

                    # 实际过滤，此时 rerank 的 top_k 应该不限制数量，或者等于当前数量
                    # rerank_strategy 对应 SearchEngine.rerank 的 mode 参数
                    remaining_records, recommended_tags = self.search_engine.rerank(
                        filter_tags=current_llm_filter_tags,
                        retrieval_records=remaining_records, # 从上一轮的结果中过滤
                        mode=rerank_strategy or "filtering",
                        top_k=len(remaining_records) # 确保只根据标签过滤，不因top_k减少数量
                    )

                    print(f"Iteration {i+1}/{max_iterations}:")
                    print(f"  LLM filter tags: {current_llm_filter_tags}")
                    print(f"  Records remaining after filtering: {len(remaining_records)}")
                    print(f"  New recommended tags: {recommended_tags[:5]}")


                    if len(remaining_records) <= rerank_stop_size:
                        print(f"Agentic search converged at iteration {i+1}. Records ({len(remaining_records)}) <= stop_size ({rerank_stop_size}).")
                        return {
                            'status': 'success',
                            'message': f"Agentic search converged after {i+1} iterations.",
                            'ranked_records': remaining_records[:rerank_stop_size], # 最终按stop_size截断
                            'filter_tags': current_llm_filter_tags
                        }
                    if not current_llm_filter_tags and len(remaining_records) > rerank_stop_size:
                        # 如果LLM不给标签，且数量仍大于阈值，可能无法收敛
                        print(f"Agentic search may not converge: No new filter tags from LLM, but {len(remaining_records)} records > stop_size {rerank_stop_size}.")
                        # 可以考虑在这里跳出，返回当前结果
                        # break

                # 循环结束但未达到 stop_size
                print(f"Agentic search reached max iterations ({max_iterations}) without converging to stop_size ({rerank_stop_size}).")
                return {
                    'status': 'fail', # 或 'success_partial'
                    'message': f"Agentic search reached max iterations. {len(remaining_records)} records remaining.",
                    'ranked_records': remaining_records[:rerank_stop_size], # 仍然截断到stop_size
                    'filter_tags': current_llm_filter_tags
                }

            except Exception as e:
                traceback.print_exc()
                return {'status': 'error', 'message': f"Error in agentic_search: {str(e)}"}

        else:
            return {'status': 'error', 'message': f"Unsupported search mode: {mode}"}

if __name__ == "__main__":
    # 示例用法：
    visitor = NaviSearchVisitor() # 假设集合已存在并加载
    search_result = visitor.search(query_text=" UPF分流策略安全机制", filter_tags=["UPF", "设备命令行/配置"]) # 标准搜索
    print(search_result)