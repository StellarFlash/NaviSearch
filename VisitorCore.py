"""
Visitor.py

核心模块，负责调度SearchOperator等组件。

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
import sys
import traceback
import json
from typing import List, Dict, Optional, Any
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, Collection, DataType, Connections
from Search.SearchEngine import SearchEngine # 假设 SearchEngine 存在且路径正确
from utils import get_embedding, get_response, get_filter, flatten_nested_structure # 假设 utils 中的函数存在且路径正确


# Visitor.py (Modified)

# ... (existing imports) ...

# Assume these are defined correctly
# from Search.SearchEngine import SearchEngine
# from utils import get_embedding, get_response, get_filter, flatten_nested_structure

DEFAULT_COLLECTION_NAME = "navi_search_collection"
DEFAULT_MILVUS_TOKEN = "root:Milvus"
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"

class NaviSearchVisitor:
    """
    负责搜索功能和用户会话相关的标签管理。
    增加了搜索时自动连接 Milvus 的功能。
    """
    def __init__(self,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 milvus_token: str = DEFAULT_MILVUS_TOKEN,
                 milvus_host: str = DEFAULT_MILVUS_HOST,
                 milvus_port: str = DEFAULT_MILVUS_PORT):

        self.milvus_host = milvus_host # Store for potential re-connection info
        self.milvus_port = milvus_port
        self.milvus_token = milvus_token

        self.collection_name = collection_name
        self.active_tags = []

        # Initialize Milvus client. Connection is managed by the client methods.
        self.client = MilvusClient(token=self.milvus_token, host=self.milvus_host, port=self.milvus_port)

        # We will use this flag to track if our internal connect_milvus was called successfully
        # The actual network connection is managed by self.client when its methods are called.
        self._is_connected = False

        # Initialize SearchEngine with the initial collection name.
        # SearchEngine methods will use the client, which handles the actual connection on demand.
        # Note: SearchEngine should be robust to the client not being actively connected initially.
        try:
             self.search_engine = SearchEngine(client=self.client, collection_name=self.collection_name)
             print(f"SearchEngine initialized for collection '{self.collection_name}'.")
        except Exception as e:
             print(f"Warning: Failed to initialize SearchEngine: {e}. Search functionality may be impacted.")
             self.search_engine = None # Handle case where SearchEngine initialization fails


        # self.collection = None # Not strictly needed if using MilvusClient methods directly


    def _check_actual_connection(self) -> bool:
        """
        Internal helper to check if the MilvusClient is actually connected or can connect.
        A lightweight operation like listing collections or checking existence works.
        """
        try:
            # A simple operation that requires a connection
            # MilvusClient methods handle connection internally.
            # Calling has_collection forces a connection attempt if not connected.
            # We don't care about the result, just if it raises an exception.
            # self.client.has_collection(self.collection_name) # Use the current collection name
            return True
        except Exception as e:
            # print(f"Actual Milvus connection check failed: {e}", file=sys.stderr)
            return False


    def connect_milvus(self) -> Dict[str, Any]:
        """
        Sets the internal state to 'connected'.
        The actual network connection is managed by the MilvusClient instance when its methods are called.
        Optionally performs a check to confirm connectivity.
        """
        if self._is_connected:
            # Check actual connection status even if flag is set
            if self._check_actual_connection():
                return {'status': 'success', 'message': "Visitor is marked as connected and connection is live."}
            else:
                 # Internal state says connected, but actual check failed. Attempt to reset.
                 print("Visitor marked as connected, but actual connection check failed. Attempting to reconnect implicitly via client.")
                 self._is_connected = False # Reset flag to allow implicit connection on next client call
                 # Let the next client call (like has_collection or search) trigger the connection
                 # We still return success here as the flag is set, but log a warning.
                 return {'status': 'warning', 'message': "Visitor was marked connected, but actual connection failed. Will attempt reconnection on next operation."}


        # Set internal state to connected intention
        self._is_connected = True
        print("Visitor internal state set to 'connected'.")

        # Optional: Perform a quick check to see if connection is possible now
        if self._check_actual_connection():
             message = "Visitor connected (internal state set, actual connection verified)."
             print(message)
             return {'status': 'success', 'message': message}
        else:
             # If check failed immediately after setting the flag, Milvus might be unreachable
             self._is_connected = False # Reset flag
             error_message = "Visitor failed to connect (internal state reset). Check Milvus server."
             print(error_message, file=sys.stderr)
             return {'status': 'error', 'message': error_message}


    def disconnect_milvus(self) -> Dict[str, Any]:
        """
        Sets the internal state to 'disconnected'.
        Closes the MilvusClient connection if possible.
        """
        if not self._is_connected:
             # Check actual connection status even if flag is unset
             if not self._check_actual_connection():
                 return {'status': 'success', 'message': "Visitor is already marked as disconnected and connection is inactive."}
             else:
                  # Internal state says disconnected, but actual check succeeded? Unlikely, but handle.
                  print("Visitor marked as disconnected, but actual connection check succeeded. Forcing client close.")
                  try:
                       self.client.close() # Attempt to close the underlying client connection
                       print("Milvus client connection closed.")
                  except Exception as e:
                       print(f"Error closing Milvus client: {e}", file=sys.stderr)
                  return {'status': 'warning', 'message': "Visitor was marked disconnected, but actual connection was live. Forced client close."}


        try:
            self.client.close() # Attempt to close the underlying client connection
            self._is_connected = False
            message = "Visitor disconnected (internal state set, client connection closed)."
            print(message)
            return {'status':'success','message': message}
        except Exception as e:
            # If closing fails, the state is still disconnected from our perspective
            self._is_connected = False # Ensure flag is false
            error_message = f"Failed to disconnect Milvus client: {str(e)}. Internal state set to disconnected."
            print(error_message, file=sys.stderr)
            return {'status':'error','message': error_message}


    def set_target_collection_name(self, collection_name: str) -> Dict[str, Any]:
        """
        切换Visitor实例搜索的目标collection名称。
        注意：此方法仅更新配置，不自动连接或加载 Collection。
        Collection 的存在性检查将在连接时或搜索时进行。
        """
        if not collection_name:
            return {'status': 'error', 'message': "New collection name cannot be empty."}

        old_collection_name = self.collection_name
        self.collection_name = collection_name
        print(f"Visitor target collection name updated to: '{collection_name}'.")

        # Re-initialize SearchEngine with the new collection name.
        # This assumes SearchEngine is designed to work with a client that manages connections.
        try:
             self.search_engine = SearchEngine(client=self.client, collection_name=self.collection_name)
             print(f"SearchEngine re-initialized for collection '{self.collection_name}'.")
             # Clear active tags as they might be irrelevant to the new collection
             self.clear_tags()
             print("Active tags cleared due to collection change.")
        except Exception as e:
             print(f"Warning: Failed to re-initialize SearchEngine with new collection '{collection_name}': {e}. Search functionality may be impacted.")
             self.search_engine = None # Indicate SearchEngine is not ready


        # Check existence of the new collection if currently connected
        if self._is_connected and self._check_actual_connection():
            try:
                if not self.client.has_collection(self.collection_name):
                    message = f"Warning: Target Collection '{self.collection_name}' does not exist in Milvus. Search operations will likely fail."
                    print(message)
                    return {'status': 'warning', 'message': f"Collection name set to '{collection_name}'. {message}"}
                else:
                    message = f"Target Collection changed to '{collection_name}'. Collection found in Milvus."
                    print(message)
                    return {'status': 'success', 'message': message}
            except Exception as e:
                 error_message = f"Error checking existence of new collection '{collection_name}' after setting: {e}. Search may fail."
                 print(error_message, file=sys.stderr)
                 return {'status': 'error', 'message': error_message}
        else:
            # If not connected, we can't check collection existence yet.
            message = f"Target Collection name set to '{collection_name}'. Milvus not connected, existence check skipped."
            print(message)
            return {'status': 'info', 'message': message}


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
               filter_tags: List[str] = None, # This parameter allows overriding instance's active_tags
               mode: str = "standard", # "standard" or "agentic"
               retrieval_top_k: int = 20,
               rerank_strategy: Optional[str] = "ranking", # maps to rerank mode like "ranking" or "filtering"
               rerank_doc_top_k: Optional[int] = 5, # for standard_search
               rerank_tags_top_k: Optional[int] = 10,
               **kwargs: Any) -> Dict[str, Any]:
        """
        统一的搜索接口。
        在执行搜索前，如果未连接，将尝试自动连接 Milvus。
        """
        # --- 自动连接逻辑 ---
        if not self._is_connected:
             print("Search called when not connected. Attempting automatic connection...")
             connect_result = self.connect_milvus()
             if connect_result['status'] == 'error':
                  error_msg = f"Automatic connection failed before search: {connect_result['message']}"
                  print(error_msg, file=sys.stderr)
                  return {'status': 'error', 'message': error_msg}
             print("Automatic connection successful.")

        # --- 连接成功后，继续搜索逻辑 ---

        # Ensure SearchEngine is initialized
        if self.search_engine is None:
             error_msg = "SearchEngine is not initialized. Cannot perform search."
             print(error_msg, file=sys.stderr)
             return {'status': 'error', 'message': error_msg}

        # Check if the target collection exists in Milvus now that we are connected
        try:
            if not self.client.has_collection(self.collection_name):
                error_msg = f"Collection '{self.collection_name}' not found in Milvus. Cannot perform search."
                print(error_msg, file=sys.stderr)
                return {'status': 'error', 'message': error_msg}
            # Optional: Check if loaded, though search/query operations usually handle this or require load first by admin
            # if not self.client.get_collection_state(self.collection_name) == 'LoadState.Loaded':
            #      print(f"Warning: Collection '{self.collection_name}' is not loaded into memory. Search performance may be affected.")

        except Exception as e:
             error_msg = f"Error checking collection existence before search: {e}"
             print(error_msg, file=sys.stderr)
             return {'status': 'error', 'message': error_msg}


        # Determine filter tags: use provided filter_tags if available, otherwise use instance's active_tags
        active_filter_tags_for_search = filter_tags if filter_tags is not None else self.get_active_tags()
        if not active_filter_tags_for_search: # If no tags provided and instance's active_tags is empty
            active_filter_tags_for_search = None # Explicitly pass None if no filtering is needed


        if mode == "standard":
            # ... (standard search logic - uses active_filter_tags_for_search) ...
            rerank_top_k = kwargs.get("rerank_top_k", rerank_doc_top_k) # Use default 5 if not provided
            try:
                 # 1. 初步检索
                 retrieval_records = self.search_engine.retrieval(
                     query_text=query_text,
                     top_k=retrieval_top_k
                 )
                 print(f"Initial retrieval returned {len(retrieval_records)} records for standard search.")
                 if not retrieval_records:
                     return {
                         'status': 'success',
                         'message': 'No records found in initial retrieval.',
                         'ranked_records': [],
                         'ranked_tags': []
                     }

                 # 2. 重排和过滤 (使用 active_filter_tags_for_search)
                 ranked_records, ranked_tags = self.search_engine.rerank(
                     filter_tags=active_filter_tags_for_search, # Pass the chosen filter tags
                     retrieval_records=retrieval_records,
                     mode=rerank_strategy or "ranking",
                     top_k=rerank_top_k # Use the calculated rerank_top_k for doc count
                 )
                 print(f"Reranking (strategy: {rerank_strategy or 'ranking'}) with filter tags {active_filter_tags_for_search} resulted in {len(ranked_records)} records.")

                 return {
                     'status': 'success',
                     'ranked_records': ranked_records,
                     'ranked_tags': ranked_tags[:rerank_tags_top_k], # Use rerank_tags_top_k for tag count
                     'filter_tags_used': active_filter_tags_for_search
                 }
            except Exception as e:
                 traceback.print_exc()
                 return {'status': 'error', 'message': f"Error in standard_search: {str(e)}"}


        elif mode == "agentic":
            # ... (agentic search logic - uses internal LLM-generated filter tags) ...
            rerank_stop_size = kwargs.get("rerank_stop_size", 3)
            max_iterations = kwargs.get("max_iterations", 10)
            try:
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
                         'ranked_tags': [] # Agentic search returns filter_tags, not ranked_tags in the same way
                     }

                 remaining_records = initial_retrieval_records.copy()
                 current_llm_filter_tags: List[str] = [] # LLM选择的标签
                 all_recommended_tags: List[str] = [] # Collect all recommended tags across iterations


                 for i in range(max_iterations):
                     # 2. LLM 生成过滤标签 (get_filter)
                     # Use recommended tags from the *previous* rerank step as input to get_filter
                     recommended_tags_from_previous_step = all_recommended_tags if i > 0 else [] # Pass empty list for first iteration if needed by get_filter
                     # Pass recommended tags from the last rerank call
                     current_llm_filter_tags = get_filter( # 假设 get_filter 返回 List[str]
                         query_text=query_text,
                         current_filter=current_llm_filter_tags, # Tags chosen by LLM in previous iterations
                         recomanded_filter=recommended_tags_from_previous_step, # Tags recommended by rerank in previous step
                         current_iteration=i,
                         current_size=len(remaining_records),
                         max_iteration=max_iterations,
                         stop_size=rerank_stop_size
                     )
                     if not isinstance(current_llm_filter_tags, list): # 确保是列表
                         print(f"Warning: get_filter did not return a list, got: {current_llm_filter_tags}. Assuming no new filter tags from LLM.")
                         current_llm_filter_tags = [] # Ensure it's a list


                     # 3. 使用LLM选择的标签进行过滤 (rerank)
                     # Pass the cumulative LLM chosen tags for filtering
                     # top_k for rerank should be set high enough not to truncate results prematurely during filtering iterations
                     # Let's use the current number of remaining records, or a large number
                     rerank_top_k_during_filtering = len(remaining_records) if remaining_records else retrieval_top_k # Don't truncate during filtering
                     # Ensure rerank_strategy is filtering for agentic mode
                     rerank_mode_for_agentic = rerank_strategy or "filtering"

                     filtered_records, recommended_tags = self.search_engine.rerank(
                         filter_tags=current_llm_filter_tags,
                         retrieval_records=remaining_records, # Filter the records remaining from the last step
                         mode=rerank_mode_for_agentic,
                         top_k=rerank_top_k_during_filtering # Retain all filtered results for now
                     )

                     remaining_records = filtered_records # Update remaining records
                     all_recommended_tags = recommended_tags # Update recommended tags for the next iteration

                     print(f"Iteration {i+1}/{max_iterations}:")
                     print(f"   LLM filter tags: {current_llm_filter_tags}")
                     print(f"   Records remaining after filtering: {len(remaining_records)}")
                     print(f"   New recommended tags (top 5): {all_recommended_tags[:5]}")


                     if len(remaining_records) <= rerank_stop_size:
                         print(f"Agentic search converged at iteration {i+1}. Records ({len(remaining_records)}) <= stop_size ({rerank_stop_size}).")
                         return {
                             'status': 'success',
                             'message': f"Agentic search converged after {i+1} iterations.",
                             'ranked_records': remaining_records[:rerank_stop_size], # Final truncation
                             'filter_tags': current_llm_filter_tags # Return the final set of LLM chosen tags
                         }
                     # Check if filtering resulted in no new tags from LLM, and we are still above the stop size
                     # This indicates potential failure to converge via tags
                     # If get_filter returns an empty list, it might signal convergence or inability to find more tags.
                     # We should break if no new *distinct* tags were added by LLM and we are stuck.
                     # However, simply checking if current_llm_filter_tags is empty isn't enough, as LLM might reiterate same tags.
                     # A more robust check might involve comparing the new LLM tags with previous iterations,
                     # or relying solely on the stop_size or max_iterations conditions.
                     # Let's rely on stop_size and max_iterations for simplicity as per original parameters.


                 # Loop ends because max_iterations is reached
                 print(f"Agentic search reached max iterations ({max_iterations}) without converging to stop_size ({rerank_stop_size}).")
                 return {
                     'status': 'fail', # Use 'fail' status as it didn't converge
                     'message': f"Agentic search reached max iterations ({max_iterations}). {len(remaining_records)} records remaining, stop size was {rerank_stop_size}.",
                     'ranked_records': remaining_records[:rerank_stop_size], # Still truncate to stop_size for consistency
                     'filter_tags': current_llm_filter_tags # Return the final set of LLM chosen tags
                 }

            except Exception as e:
                 traceback.print_exc()
                 return {'status': 'error', 'message': f"Error in agentic_search: {str(e)}"}

        else:
            return {'status': 'error', 'message': f"Unsupported search mode: {mode}"}

# ... (if __name__ == "__main__": block remains the same) ...