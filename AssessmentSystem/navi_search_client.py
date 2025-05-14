import os
import dotenv
import requests
import json
from typing import List, Dict, Optional, Any

from AssessmentSystem.model import AssessmentSpecItem, EvidenceMaterial, EvidenceSearchParams, EvidenceSearchResult
from AssessmentSystem.evidence_loader import EvidenceLoader
from AssessmentSystem.llm_client import LLMAssessmentClient

dotenv.load_dotenv()

evidence_loader = EvidenceLoader()

class NaviSearchClient:
    def __init__(self, admin_url: str, visitor_url: str, evidence_collection_name: str, llm_client:LLMAssessmentClient):
        self.admin_url = admin_url
        self.visitor_url = visitor_url
        self.evidence_collection_name = evidence_collection_name
        self.llm_client = llm_client # Store the LLM client instance
        # 尝试连接 Visitor 服务
        self._set_collection(evidence_collection_name)
        self._connect()
        # 初始化集合（创建或使用现有）
        self.init_collection(evidence_collection_name)

        try:
            # 加载本地证据文件
            evidences = evidence_loader.load_evidences("AssessmentSystem/evidences.jsonl")
            if evidences:
                # 插入所有加载的证据到集合
                self._insert_evidences(evidence_collection_name, evidences)
            else:
                print("没有加载到证据，跳过插入。")
        except FileNotFoundError:
            print("错误：找不到 AssessmentSystem/evidences.jsonl 文件，跳过证据插入。")
        except Exception as e:
            print(f"加载或插入证据时发生错误: {e}")

    # def __del__(self):
    #     # 在对象销毁时断开 Visitor 服务连接
    #     self._disconnect()

    def _set_collection(self, collection_name: str):
        """
        调用 Visitor API 的 /set_collection 接口设置当前集合。
        """
        url = f"{self.visitor_url}/set_collection"
        payload = {"collection_name": collection_name}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() # 对不良状态码抛出 HTTPError
            print(f"当前集合已设置为: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"设置集合时发生错误: {e}")
            # 根据需要处理错误
    def _connect(self):
        """
        调用 Visitor API 的 /connect 接口连接 Milvus。
        """
        url = f"{self.admin_url}/connect"
        try:
            response = requests.post(url)
            # response.raise_for_status() # 对不良状态码抛出 HTTPError
            print(f"Visitor 服务连接成功: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"连接 Visitor 服务时发生错误: {e}")
            # 根据需要处理连接错误，例如重试或退出


    def _disconnect(self):
        """
        调用 Visitor API 的 /disconnect 接口断开 Milvus 连接。
        """
        url = f"{self.visitor_url}/disconnect"
        try:
            response = requests.post(url)
            response.raise_for_status() # 对不良状态码抛出 HTTPError
            print(f"Visitor 服务断开连接: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"断开 Visitor 服务连接时发生错误: {e}")
            # 根据需要处理断开连接错误

    def init_collection(self, evidence_collection_name: str):
        """
        调用 Admin API 的 /collections/init 接口初始化集合。
        """
        url = f"{self.admin_url}/collections/init"
        payload = {
            "collection_name": evidence_collection_name,
            "drop_existing": True, # 示例：如果集合已存在则删除
            # 可选：在这里添加自定义的 custom_schema 和 index_params
            "custom_schema": None, # 使用默认 Schema
            "index_params": None # 使用默认索引参数
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() # 对不良状态码抛出 HTTPError
            print(f"集合 '{evidence_collection_name}' 初始化结果: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"初始化集合 '{evidence_collection_name}' 时发生错误: {e}")
            # 根据需要处理错误

    def _insert_evidences(self, evidence_collection_name: str, evidences: List[EvidenceMaterial]):
        """
        调用 Admin API 的 /records/insert_many 接口批量插入证据。
        """
        url = f"{self.admin_url}/records/insert_many"
        records_data = []
        for evidence in evidences:
            # 将 EvidenceMaterial 对象转换为适合 API 请求的字典
            record_data = {
                "content": evidence.content,
                "tags": evidence.tags,
                "embedding": evidence.embedding,
                # 添加 Schema 中定义的其他字段
            }
            records_data.append(record_data)

        payload = {
            "collection_name": evidence_collection_name,
            "records": records_data,
            "auto_generate_embedding": True # 假设服务会自动生成 Embedding
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() # 对不良状态码抛出 HTTPError
            print(f"已向集合 '{evidence_collection_name}' 插入 {len(records_data)} 条记录: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"向集合 '{evidence_collection_name}' 插入记录时发生错误: {e}")
            # 处理错误

    def _search(self, query: str, tags: List[str], mode: str = "standard") -> Dict:
        """
        调用 Visitor API 的 /search 接口执行搜索。
        """
        url = f"{self.visitor_url}/search"
        payload = {
            "query_text": query,
            "filter_tags": tags,
            "mode": mode,
            # 根据 VisitorFastAPI.PY 的 SearchRequest 模型添加其他搜索参数
            "retrieval_top_k": 20, # 示例值
            "rerank_strategy": "ranking",# 示例值
            "rerank_top_k_standard": 5, # 示例值
            "max_iterations_agentic": 10
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() # 对不良状态码抛出 HTTPError
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"搜索时发生错误: {e}")
            return {"status": "error", "message": str(e)}

    def _get_source_from_record(self, record: Dict[str, Any]) -> str:
        """Helper function to safely get and stringify source from a record."""
        source_val = record.get("source")
        if source_val is not None:
            return str(source_val) # Ensure string if source field exists

        id_val = record.get("id")
        if id_val is not None:
            return str(id_val) # Convert ID to string

        return "unknown" # Default if neither source nor id is found

    def _search_via_visitor_api(self, query_text: str, filter_tags: List[str], mode: str = "standard") -> Dict[str, Any]:
        """
        调用 Visitor API 的 /search 接口执行单次搜索。
        现在总是以 "standard" 模式调用，因为迭代逻辑由 LLMAssessmentClient 处理。
        """
        url = f"{self.visitor_url}/search"
        payload = {
            "query_text": query_text,
            "filter_tags": filter_tags,
            "mode": mode, # Should be "standard" as agentic logic is now in LLMClient
            "retrieval_top_k": 20,
            "rerank_strategy": "ranking",
            "rerank_top_k_standard": 10, # Get more docs for LLM to choose from if needed for context
            "rerank_tags_top_k": 10 # Get relevant tags
            # Agentic specific params from Visitor API are not used here anymore
        }
        print(f"  调用 Visitor API 搜索: Query='{query_text[:50]}...', Tags={filter_tags}, Mode={mode}")
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Visitor API 搜索时发生错误: {e}")
            return {"status": "error", "message": str(e), "ranked_records": [], "ranked_tags": []}
        except json.JSONDecodeError:
            print(f"  Visitor API 搜索时返回非JSON响应: {response.text}")
            return {"status": "error", "message": "Visitor API returned non-JSON response", "ranked_records": [], "ranked_tags": []}

    # def search_evidence(self, search_params: EvidenceSearchParams) -> List[EvidenceSearchResult]:
    #     """
    #     根据评估规范搜索证据。

    #     Args:
    #         assessment_spec: 评估规范项，包含内容和评估方法

    #     Returns:
    #         证据搜索结果列表，已转换为标准格式
    #     """
    #     query = search_params.query_text  # 从 assessment_spec 提取查询文本
    #     tags = search_params.filter_tags  # 从 assessment_spec 提取标签

    #     try:
    #         search_result = self._search(query, tags, mode="agentic")

    #         evidence_results: List[EvidenceSearchResult] = []

    #         if not search_result:
    #             return evidence_results

    #         # 处理不同状态的结果
    #         if search_result.get("status") == "success":
    #             # 核心模块返回的是ranked_records而不是results
    #             for record in search_result.get("ranked_records", []):
    #                 try:
    #                     # 将记录映射到EvidenceSearchResult
    #                     evidence = EvidenceSearchResult(
    #                         source=record.get("source", "unknown"),
    #                         content=record.get("content", ""),
    #                         # 其他额外字段
    #                     )
    #                     evidence_results.append(evidence)
    #                 except Exception as e:
    #                     print(f"处理搜索记录时发生错误: {record}. 错误: {e}")
    #                     continue

    #         elif search_result.get("status") in ["fail", "error"]:
    #             print(f"搜索失败: {search_result.get('message', '未知错误')}")
    #             # 可以选择返回空列表或部分结果

    #         return evidence_results

    #     except Exception as e:
    #         print(f"搜索过程中发生异常: {e}")
    #         return []

    def search_evidence(self, spec_item: AssessmentSpecItem) -> List[EvidenceSearchResult]:
        """
        根据评估规范迭代搜索证据。
        它使用 LLMAssessmentClient 来生成和优化搜索参数。

        Args:
            spec_item: 评估规范项。

        Returns:
            最终找到的证据搜索结果列表。
        """
        current_query: Optional[str] = None
        current_tags: Optional[List[str]] = None
        ranked_docs_for_llm: Optional[List[EvidenceSearchResult]] = None
        ranked_tags_for_llm: Optional[List[str]] = None

        final_ranked_records_from_api: List[Dict[str, Any]] = []

        for i in range(self.llm_client.max_search_iterations):
            print(f"\n开始证据搜索迭代 {i + 1}/{self.llm_client.max_search_iterations} for spec ID: {spec_item.id}")

            search_params_for_iteration = self.llm_client.generate_search_params(
                spec_item=spec_item,
                iteration=i,
                current_query_text=current_query,
                current_filter_tags=current_tags,
                ranked_docs=ranked_docs_for_llm,
                ranked_tags=ranked_tags_for_llm
            )

            current_query = search_params_for_iteration.query_text
            current_tags = search_params_for_iteration.filter_tags
            print(f"  迭代 {i+1} 使用参数: Query='{current_query[:100]}...', Tags={current_tags}, Terminated={search_params_for_iteration.terminated}")

            raw_search_result = self._search_via_visitor_api(
                query_text=current_query,
                filter_tags=current_tags,
                mode="standard"
            )
            evidence_search_params = EvidenceSearchParams(query_text= current_query, filter_tags = current_tags)
            if raw_search_result.get("status") == "success":
                final_ranked_records_from_api = raw_search_result.get("ranked_records", [])

                ranked_docs_for_llm = [
                    EvidenceSearchResult(
                        source=self._get_source_from_record(record), # Use helper
                        content=record.get("content", ""),
                        tags=record.get("tags", [])
                    ) for record in final_ranked_records_from_api
                ]
                ranked_tags_for_llm = raw_search_result.get("ranked_tags", [])
                print(f"  迭代 {i+1} 搜索成功: 返回 {len(final_ranked_records_from_api)} 条记录, {len(ranked_tags_for_llm)} 个推荐标签。")
            else:
                print(f"  迭代 {i+1} 搜索失败: {raw_search_result.get('message', '未知错误')}")
                if i == 0:
                    return []
                else:
                    break

            if search_params_for_iteration.terminated:
                print(f"  搜索参数生成器指示在迭代 {i + 1} 后终止。")
                break

            if (i + 1) >= self.llm_client.max_search_iterations:
                 print(f"  已达到最大搜索迭代次数 {self.llm_client.max_search_iterations}。")
                 break

        evidence_results: List[EvidenceSearchResult] = []
        if final_ranked_records_from_api:
            for record_dict in final_ranked_records_from_api:
                evidence_results.append(
                    EvidenceSearchResult(
                        source=self._get_source_from_record(record_dict), # Use helper
                        content=record_dict.get("content", ""),
                        tags=record_dict.get("tags", [])
                    )
                )
        print(f"迭代搜索完成。最终返回 {len(evidence_results)} 条证据。")
        return evidence_results, evidence_search_params

# 示例用法（假设您的 FastAPI 服务正在运行）
if __name__ == "__main__":
    # 替换为您实际运行的 Admin 和 Visitor FastAPI 服务的 URL
    admin_api_url = "http://" + os.getenv("ADMIN_API_HOST") + ":" + os.getenv("ADMIN_API_PORT")
    visitor_api_url = "http://" + os.getenv("VISITOR_API_HOST") + ":" + os.getenv("VISITOR_API_PORT")
    collection_name = "my_evidence_collection"

    print(f"尝试连接 Admin ({admin_api_url})")
    llm_client = LLMAssessmentClient()
    # 初始化客户端，这也会初始化集合并插入数据
    client = NaviSearchClient(admin_api_url, visitor_api_url, collection_name, llm_client = llm_client)

    # 示例搜索
    assessment_item = AssessmentSpecItem(id = "abcd", condition = "required",heading = "", content="All sensitive data is masked in application logs.", method="查阅安全规范文档")
    print(f"\n搜索评估证据: {assessment_item.content}")

    found_evidence = client.search_evidence(assessment_item)

    if found_evidence:
        print("\n找到的证据结果:")
        for i, evidence in enumerate(found_evidence):
            print(f"结果 {i+1}:")
            print(f"  内容: {evidence.content[:200]}...") # 打印内容前200字符
            print("-" * 20)
    else:
        print("\n未找到证据。")
    exit()
    # 当 client 对象被垃圾回收或脚本退出时，会自动调用 __del__ 方法。
    # 您也可以显式删除对象：
    del client