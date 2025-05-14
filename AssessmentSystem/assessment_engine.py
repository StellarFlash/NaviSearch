import os
import dotenv
import json
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 从您的模型、加载器和客户端模块导入必要的类和枚举
from AssessmentSystem.model import (
    AssessmentSpecItem,
    AssessmentResult,
    AssessmentReport,
    Judgement,
    AssessmentStatus,
    Conclusion,
    EvidenceSearchParams,
    EvidenceSearchResult # 导入EvidenceSearchResult，因为worker返回的结果中包含它
)
from AssessmentSystem.spec_loader import SpecLoader
# pylint: disable=import-error,no-name-in-module
from AssessmentSystem.assessment_worker import AssessmentWorker
from AssessmentSystem.llm_client import LLMAssessmentClient
from AssessmentSystem.navi_search_client import NaviSearchClient

dotenv.load_dotenv() # 加载环境变量

class AssessmentEngine:
    """
    评估引擎，负责加载评估规范，启动 AssessmentWorker，采集评估结果，并导出报告。
    """
    def __init__(
        self,
        spec_file_path: str,
        admin_api_url: str,
        visitor_api_url: str,
        evidence_collection_name: str,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model_name: str = "qwen-plus-latest",
        llm_temperature: float = 0.0,
        worker_timeout_seconds: int = 600, # 设置每个 worker 的超时时间
        max_workers: int = 5 # 控制并发 worker 数量
    ):
        """
        初始化 AssessmentEngine。

        Args:
            spec_file_path: 评估规范 JSONL 文件的路径。
            admin_api_url: NaviSearch Admin API 的 URL。
            visitor_api_url: NaviSearch Visitor API 的 URL。
            evidence_collection_name: NaviSearch 中用于存储证据的集合名称。
            llm_base_url: LLM API 的基础 URL (如果未设置环境变量)。
            llm_api_key: LLM API 密钥 (如果未设置环境变量)。
            llm_model_name: 使用的 LLM 模型名称。
            llm_temperature: LLM 生成的温度参数。
            worker_timeout_seconds: 单个 AssessmentWorker 的处理超时时间（秒）。
            max_workers: 并发执行的 AssessmentWorker 数量。
        """
        self.spec_file_path = spec_file_path
        self.admin_api_url = admin_api_url
        self.visitor_api_url = visitor_api_url
        self.evidence_collection_name = evidence_collection_name
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.worker_timeout_seconds = worker_timeout_seconds
        self.max_workers = max_workers

        self.spec_loader = SpecLoader()
        self.assessment_results: List[AssessmentResult] = []

    def run_assessment(self) -> AssessmentReport:
        """
        运行整个评估流程：加载规范 -> 遍历并启动 worker -> 收集结果。

        Returns:
            生成的评估报告。
        """
        print(f"开始加载评估规范文件: {self.spec_file_path}")
        try:
            spec_items = self.spec_loader.load_specs(self.spec_file_path)
            print(f"成功加载 {len(spec_items)} 条评估规范。")
        except FileNotFoundError:
            print(f"错误：找不到评估规范文件: {self.spec_file_path}")
            # 返回一个包含错误信息的报告，或者抛出异常
            report = AssessmentReport()
            report.assessment_results.append(
                 AssessmentResult(
                     spec_id="N/A",
                     spec_content=f"加载规范文件失败: {self.spec_file_path}",
                     status=AssessmentStatus.FAIL,
                     error_message=f"FileNotFoundError: {self.spec_file_path}"
                 )
            )
            report.statics[AssessmentStatus.FAIL] = 1 # 更新统计
            return report
        except Exception as e:
            print(f"加载评估规范时发生意外错误: {e}")
            report = AssessmentReport()
            report.assessment_results.append(
                 AssessmentResult(
                     spec_id="N/A",
                     spec_content=f"加载规范文件时发生错误",
                     status=AssessmentStatus.FAIL,
                     error_message=f"加载错误: {str(e)}"
                 )
            )
            report.statics[AssessmentStatus.FAIL] = 1
            return report

        if not spec_items:
            print("没有找到评估规范，评估结束。")
            return AssessmentReport() # 返回一个空的报告

        print("初始化 LLM 客户端...")
        try:
            llm_client = LLMAssessmentClient(
                llm_base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                model_name=self.llm_model_name,
                temperature=self.llm_temperature
            )
            print("LLM 客户端初始化成功。")
        except ValueError as e:
             print(f"LLM 客户端初始化失败: {e}")
             # 记录所有规范为失败
             failed_results = [
                AssessmentResult(
                    spec_id=item.id,
                    spec_content=item.content,
                    status=AssessmentStatus.FAIL,
                    error_message=f"LLM 客户端初始化失败: {str(e)}"
                ) for item in spec_items
            ]
             report = AssessmentReport(assessment_results=failed_results)
             report.statics[AssessmentStatus.FAIL] = len(spec_items)
             return report

        # 初始化客户端，确保它们在 worker 之前可用
        print("初始化 NaviSearch 客户端...")
        try:
            navisearch_client = NaviSearchClient(
                admin_url=self.admin_api_url,
                visitor_url=self.visitor_api_url,
                evidence_collection_name=self.evidence_collection_name,
                llm_client = llm_client
            )
            print("NaviSearch 客户端初始化成功。")
        except Exception as e:
            print(f"NaviSearch 客户端初始化失败: {e}")
            # 记录所有规范为失败
            failed_results = [
                AssessmentResult(
                    spec_id=item.id,
                    spec_content=item.content,
                    status=AssessmentStatus.FAIL,
                    error_message=f"NaviSearch 客户端初始化失败: {str(e)}"
                ) for item in spec_items
            ]
            report = AssessmentReport(assessment_results=failed_results)
            report.statics[AssessmentStatus.FAIL] = len(spec_items)
            return report



        print(f"开始执行评估任务，并发 worker 数量: {self.max_workers}")
        self.assessment_results = [] # 清空之前的评估结果

        # 使用 ThreadPoolExecutor 实现并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务到线程池
            future_to_spec_id = {
                executor.submit(
                    AssessmentWorker(
                        navisearch_client=navisearch_client,
                        llm_client=llm_client,
                        timeout_seconds=self.worker_timeout_seconds
                    ).process_task,
                    spec_item
                ): spec_item.id for spec_item in spec_items
            }

            # 收集任务结果
            for future in as_completed(future_to_spec_id):
                spec_id = future_to_spec_id[future]
                try:
                    worker_result_dict = future.result()
                    # 将 worker 返回的字典转换为 AssessmentResult Pydantic 模型
                    assessment_result = self._parse_worker_result(worker_result_dict)
                    self.assessment_results.append(assessment_result)
                    print(f"规范 {spec_id} 处理完成，状态: {assessment_result.status}")
                except Exception as e:
                    print(f"规范 {spec_id} 处理过程中发生未捕获的异常: {e}")
                    # 如果 worker 内部没有完全捕获错误，这里会处理
                    self.assessment_results.append(
                        AssessmentResult(
                            spec_id=spec_id,
                            spec_content="加载规范内容失败或处理异常", # 尝试获取原始内容，如果可以
                            evidence_search_params = EvidenceSearchParams(query_text = "", filter_tags = []),
                            status=AssessmentStatus.FAIL,
                            error_message=f"未捕获的异常: {str(e)}"
                        )
                    )

        print("所有评估任务执行完毕。")
        return self.generate_report()

    def _parse_worker_result(self, worker_result_dict: Dict[str, Any]) -> AssessmentResult:
        """
        将 AssessmentWorker 返回的字典结果转换为 AssessmentResult Pydantic 模型。
        此版本修正了对 evidence 列表的处理。
        """
        try:
            # 1. 直接获取顶层的 evidence 列表
            # 假设 worker_result_dict["evidence"] 已经是 List[EvidenceSearchResult]
            # 或者在出错时是 List[Dict] (例如，如果LLM返回了原始字典而不是对象)
            # evidence_from_worker = worker_result_dict.get("evidence", [])

            # parsed_evidence_list: List[EvidenceSearchResult] = []
            # if isinstance(evidence_from_worker, list):
            #     for item in evidence_from_worker:
            #         if isinstance(item, EvidenceSearchResult): # 如果已经是 EvidenceSearchResult 对象
            #             parsed_evidence_list.append(item)
            #         elif isinstance(item, dict): # 如果是字典，尝试实例化（作为一种兼容或错误恢复）
            #             try:
            #                 # 确保这里的 EvidenceSearchResult 模型定义与 LLMClient 中的一致
            #                 # 特别是如果它包含如 'tags' 等额外字段
            #                 parsed_evidence_list.append(EvidenceSearchResult(**item))
            #             except Exception as e_instantiate:
            #                 print(f"警告: 无法将字典实例化为 EvidenceSearchResult: {item}, 错误: {e_instantiate}")
            #         else:
            #             print(f"警告: 'evidence' 列表中包含意外类型: {type(item)}")
            # else:
            #     print(f"警告: worker_result_dict 中的 'evidence' 不是列表，实际类型: {type(evidence_from_worker)}")
            # 1. 解析 evidence 搜索阐述
            evidence_search_parmas = worker_result_dict.get("evidence_search_parmas")
            # 2. 解析 conclusion 字典中的 judgement 和 comment
            conclusion_data = worker_result_dict.get("conclusion", {})
            judgement_str = conclusion_data.get("judgement", Judgement.NOT_PROCESSED.value).strip().lower()
            try:
                judgement = Judgement(judgement_str)
            except ValueError:
                print(f"警告: 规范 {worker_result_dict.get('spec_id', '未知')} LLM 返回了无效的 judgement: '{judgement_str}'。将设为 ERROR。")
                judgement = Judgement.ERROR

            comment = conclusion_data.get("comment", "")

            # 3. 构建 Conclusion 对象，并传入已解析的证据列表
            conclusion_obj = Conclusion(
                judgement=judgement,
                comment=comment,
                evidence=worker_result_dict.get("evidence", []) # 直接使用 worker 返回的 evidence 列表
            )

            # 4. 解析状态和错误信息
            status_str = worker_result_dict.get("status", "fail").lower()
            try:
                status = AssessmentStatus(status_str)
                error_message = worker_result_dict.get("error_message")
                # 如果worker成功，但LLM未能返回有效结论或有效证据索引，可能需要调整最终状态
                if status == AssessmentStatus.SUCCESS:
                    if conclusion_obj.judgement == Judgement.NOT_PROCESSED or conclusion_obj.judgement == Judgement.ERROR:
                        # status = AssessmentStatus.FAIL # 或者保持 SUCCESS 但依赖 comment
                        if not error_message: # 如果没有更具体的错误信息
                            error_message = conclusion_obj.comment # 使用 conclusion 的 comment 作为错误提示
            except ValueError:
                status = AssessmentStatus.FAIL
                error_message = f"AssessmentWorker 返回了无效的状态: {status_str}"

            # 5. 构建并返回 AssessmentResult 对象
            return AssessmentResult(
                spec_id=worker_result_dict.get("spec_id", "未知ID"),
                spec_content=worker_result_dict.get("spec_content", "未知内容"),
                evidence_search_params = evidence_search_parmas,
                conclusion=conclusion_obj,
                status=status,
                error_message=error_message
            )
        except Exception as e:
            print(f"解析 AssessmentWorker 结果时发生严重错误: {worker_result_dict}. 错误: {e}")
            # 返回一个表示解析失败的 AssessmentResult
            return AssessmentResult(
                spec_id=worker_result_dict.get("spec_id", "解析错误"),
                spec_content=worker_result_dict.get("spec_content", "内容解析错误"),
                evidence_search_params = evidence_search_parmas,
                status=AssessmentStatus.FAIL,
                conclusion=Conclusion(judgement=Judgement.ERROR, comment=f"引擎解析worker结果失败: {str(e)}", evidence=[]),
                error_message=f"引擎解析worker结果失败: {str(e)}"
            )

    def generate_report(self) -> AssessmentReport:
        """
        根据收集到的评估结果生成评估报告。

        Returns:
            一个 AssessmentReport 对象。
        """
        report = AssessmentReport(assessment_results=self.assessment_results)

        # 计算统计信息
        for result in self.assessment_results:
            if result.status == AssessmentStatus.SUCCESS:
                 # 只有成功完成的才统计结论
                 if result.conclusion.judgement == Judgement.COMPLIANT:
                     report.statics["compliant"] += 1
                 elif result.conclusion.judgement == Judgement.NON_COMPLIANT:
                     report.statics["non_compliant"] += 1
                 elif result.conclusion.judgement == Judgement.NOT_APPLICABLE:
                     report.statics["not_applicable"] += 1
                 else:
                     # 理论上成功状态下不应该出现 NOT_PROCESSED，但作为容错处理
                     report.statics["not_processed"] += 1
            elif result.status == AssessmentStatus.TIMEOUT:
                 report.statics["timeout"] = report.statics.get("timeout", 0) + 1
            elif result.status == AssessmentStatus.FAIL:
                 report.statics["fail"] = report.statics.get("fail", 0) + 1
            # RUNNING 状态不应该在最终报告中出现，如果出现说明流程有问题

        return report

    def export_report(self, report: AssessmentReport, output_file_path: str):
        """
        将评估报告导出为 JSON 文件。

        Args:
            report: 要导出的评估报告对象。
            output_file_path: 报告输出的文件路径。
        """
        try:
            # 使用 model 的 json() 方法方便地序列化 Pydantic 对象
            report_json = report.model_dump_json(indent=4)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(report_json)
            print(f"评估报告已成功导出到: {output_file_path}")
        except Exception as e:
            print(f"导出评估报告时发生错误: {e}")

# 示例用法
if __name__ == "__main__":
    # 请根据您的实际环境配置这些参数
    SPEC_FILE = "AssessmentSystem/assessment_items.jsonl"  # 评估规范文件路径
    REPORT_FILE = "AssessmentSystem/assessment_report.json" # 评估报告输出路径
    NAVISEARCH_ADMIN_URL = os.getenv("ADMIN_API_HOST") + ":" + os.getenv("ADMIN_API_PORT") # NaviSearch Admin API 地址
    NAVISEARCH_VISITOR_URL = os.getenv("VISITOR_API_HOST") + ":" + os.getenv("VISITOR_API_PORT") # NaviSearch Visitor API 地址
    EVIDENCE_COLLECTION = "security_assessment_evidence" # NaviSearch 集合名称
    # LLM 配置，如果环境变量已设置，可以不在这里指定
    LLM_BASE_URL = None # 或 "http://your_llm_api_host:port"
    LLM_API_KEY = None # 或 "your_api_key"
    LLM_MODEL = "qwen-plus-latest" # 或您使用的模型名称

    print("正在初始化评估引擎...")
    try:
        engine = AssessmentEngine(
            spec_file_path=SPEC_FILE,
            admin_api_url=NAVISEARCH_ADMIN_URL,
            visitor_api_url=NAVISEARCH_VISITOR_URL,
            evidence_collection_name=EVIDENCE_COLLECTION,
            llm_base_url=LLM_BASE_URL,
            llm_api_key=LLM_API_KEY,
            llm_model_name=LLM_MODEL,
            worker_timeout_seconds=600, # 单个 worker 超时 10 分钟
            max_workers=5 # 5 个并发 worker
        )
        print("评估引擎初始化成功。")

        print("正在运行评估流程...")
        final_report = engine.run_assessment()
        print("评估流程完成。")

        # 打印评估统计摘要
        print("\n--- 评估统计摘要 ---")
        for status, count in final_report.statics.items():
             print(f"{status}: {count}")
        print("--------------------\n")


        print("正在导出评估报告...")
        engine.export_report(final_report, REPORT_FILE)

    except Exception as e:
        print(f"评估引擎运行过程中发生致命错误: {e}")