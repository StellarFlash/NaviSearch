# llm_client.py
import time
import random
import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import requests
from pydantic import ValidationError
import openai
from openai import OpenAI
# Import data models from AssessmentSystem.model
from AssessmentSystem.model import AssessmentSpecItem, EvidenceSearchParams, EvidenceSearchResult, Judgement, Conclusion


load_dotenv()

class LLMAssessmentClient:
    """
    一个与 LLM 服务交互的客户端。
    使用 OpenAI 风格的 API。
    """
    def __init__(self, llm_base_url: str = None, api_key: str = None, model_name: str = "qwen-plus-latest", temperature: float = 0.0, max_search_iterations = 5):

        # 如果没有显式提供，则从环境变量加载
        self.base_url = llm_base_url or os.getenv("LLM_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model_name = model_name or os.getenv("LLM_MODEL")
        self.temperature = temperature # Add temperature as an initialization parameter
        self.max_search_iterations = max_search_iterations # Add max_search_iterations as an initialization parameter

        if not self.base_url:
            raise ValueError("LLM_BASE_URL 必须设置，可以作为参数或在 .env 文件中设置")
        if not self.api_key:
            print("警告：没有 API 密钥，LLM 服务可能无法工作！") # Optional: Warning if no key provided
        if not self.model_name:
           raise ValueError("LLM_MODEL 必须设置，可以作为参数或在 .env 文件中设置")

        print(f"LLMAssessmentClient 初始化，API URL: {self.base_url}, 模型: {self.model_name}")

        self.client = OpenAI(base_url = self.base_url,api_key = self.api_key) # Initialize OpenAI client with base_url and api_key (if provided)are headers for API calls (if using an API key)

    def generate_search_params(
        self,
        spec_item: AssessmentSpecItem,
        iteration: int,
        current_query_text: Optional[str] = None,
        current_filter_tags: Optional[List[str]] = None,
        ranked_docs: Optional[List[EvidenceSearchResult]] = None,
        ranked_tags: Optional[List[str]] = None
    ) -> EvidenceSearchParams:
        """
        生成或优化证据搜索参数，支持迭代。
        在第一次迭代时，它基于评估规范直接构造查询。
        在后续迭代中，它调用LLM根据先前的搜索结果（ranked_docs, ranked_tags）和当前参数来优化查询和过滤标签。

        Args:
            spec_item: 当前评估规范条目。
            iteration: 当前迭代次数 (0-based)。
            current_query_text: 上一次迭代使用的查询文本 (用于后续迭代)。
            current_filter_tags: 上一次迭代使用的过滤标签 (用于后续迭代)。
            ranked_docs: 上一次搜索返回的排序后的文档列表。
            ranked_tags: 上一次搜索返回的推荐标签列表。

        Returns:
            一个 EvidenceSearchParams 对象，包含新的查询文本、过滤标签和终止状态。
        """
        print(f"生成搜索参数 - 迭代: {iteration + 1}/{self.max_search_iterations}")

        if iteration == 0:
            # 第一次迭代：直接使用规范内容构建查询，无过滤标签
            initial_query_text = f"{spec_item.content}\n评估方法: {spec_item.method}"
            # 第一次迭代后是否终止取决于最大迭代次数
            terminated_after_first = self.max_search_iterations <= 1
            print(f"  首次迭代，生成初始查询参数。终止状态: {terminated_after_first}")
            return EvidenceSearchParams(
                query_text=initial_query_text,
                filter_tags=[],
                terminated=terminated_after_first
            )

        # 后续迭代或达到最大迭代次数前的最后一次机会调用LLM
        if iteration >= self.max_search_iterations:
            print(f"  已达到最大迭代次数 ({self.max_search_iterations})。终止搜索。")
            return EvidenceSearchParams(
                query_text=current_query_text or f"{spec_item.content}\n评估方法: {spec_item.method}", # Fallback if current_query_text is None
                filter_tags=current_filter_tags or [],
                terminated=True
            )

        # --- 为LLM构建Prompt (迭代 > 0) ---
        prompt_template_iterative = """
你是一个协助进行网络安全评估证据检索的AI助手。
请根据以下评估规范、当前的查询参数、以及上一轮检索到的文档和推荐标签，生成一组更优化的查询参数，以帮助找到最相关的证据。

评估规范条目ID: {spec_item_id}
原始评估规范内容:
{spec_item_content}
原始评估方法:
{spec_item_method}

当前迭代的上下文信息:
当前查询文本: {current_query_text}
当前已选过滤标签: {current_filter_tags_str}

上一轮检索到的主要文档摘要 (最多5条):
{formatted_ranked_docs}

上一轮推荐的相关标签 (最多10条):
{formatted_ranked_tags}

你的任务是：
1.  分析以上所有信息。
2.  决定是否需要优化当前的“查询文本”。如果需要，提供一个新的“new_query_text”。如果不需要，可以返回当前的查询文本或留空。
3.  根据所有信息，建议一组新的“过滤标签 (new_filter_tags)”列表。这些标签应有助于更精确地定位相关证据。你可以从推荐标签中选择，也可以生成新的、更具体的标签。如果不需要新的过滤标签，可以返回空列表。
4.  判断是否已经找到了足够精确的参数，或者是否应该终止迭代搜索 ("terminate_search": true/false)。如果认为当前的文档和标签已经足够好，或者进一步优化意义不大，则应终止。

输出格式必须为JSON，包含以下字段:
"new_query_text": "优化后的查询文本，如果无优化则为空字符串或当前文本",
"new_filter_tags": ["新的过滤标签列表"],
"terminate_search": true  // 或 false
"""
        formatted_ranked_docs_str = "无相关文档。"
        if ranked_docs:
            doc_summaries = []
            for i, doc in enumerate(ranked_docs[:5]): # 最多显示前5个文档
                content_preview = doc.content
                if len(doc.content) > 200:
                    content_preview = doc.content[:200] + "..."
                doc_summaries.append(f"  文档 {i+1} (来源: {doc.source}): {content_preview}\n    标签: {doc.tags if hasattr(doc, 'tags') and doc.tags else '无'}")
            formatted_ranked_docs_str = "\n".join(doc_summaries)

        formatted_ranked_tags_str = "无推荐标签。"
        if ranked_tags:
            formatted_ranked_tags_str = ", ".join(ranked_tags[:10]) # 最多显示前10个推荐标签

        prompt = prompt_template_iterative.format(
            spec_item_id=spec_item.id,
            spec_item_content=spec_item.content,
            spec_item_method=spec_item.method,
            current_query_text=current_query_text or "无 (请基于规范内容生成)",
            current_filter_tags_str=str(current_filter_tags) if current_filter_tags else "无",
            formatted_ranked_docs=formatted_ranked_docs_str,
            formatted_ranked_tags=formatted_ranked_tags_str
        )
        # print(f"  迭代 {iteration+1} Prompt 发送给 LLM:\n{prompt[:500]}...") # 打印部分Prompt进行调试

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature, # Use a slightly higher temperature for creative refinement
                response_format={"type": "json_object"} if "qwen" in self.model_name else None
            )

            llm_response_content = response.choices[0].message.content
            if not llm_response_content:
                print("  LLM返回了空响应，使用当前参数并终止。")
                return EvidenceSearchParams(
                    query_text=current_query_text or f"{spec_item.content}\n评估方法: {spec_item.method}",
                    filter_tags=current_filter_tags or [],
                    terminated=True
                )

            llm_params = json.loads(llm_response_content)

            new_query = llm_params.get("new_query_text", "").strip()
            if not new_query: # 如果LLM返回空查询文本，则沿用上一次的或初始的
                new_query = current_query_text or f"{spec_item.content}\n评估方法: {spec_item.method}"

            new_tags = llm_params.get("new_filter_tags", [])
            if not isinstance(new_tags, list): # 确保是列表
                print(f"  LLM返回的new_filter_tags不是列表: {new_tags}。将使用空标签列表。")
                new_tags = []

            terminate_flag = llm_params.get("terminate_search", False)

            # 如果LLM没有要求终止，但这是允许的最大迭代次数前的最后一次调用，则强制终止
            if not terminate_flag and (iteration + 1) >= self.max_search_iterations:
                terminate_flag = True
                print(f"  LLM未要求终止，但已达到最大迭代次数前的最后一次调用。强制终止。")

            print(f"  LLM生成的新参数: Query='{new_query[:100]}...', Tags={new_tags}, Terminate={terminate_flag}")
            return EvidenceSearchParams(query_text=new_query, filter_tags=new_tags, terminated=terminate_flag)

        except json.JSONDecodeError as e:
            print(f"  无法解析LLM的JSON响应: {str(e)}。使用当前参数并终止。")
            return EvidenceSearchParams(
                query_text=current_query_text or f"{spec_item.content}\n评估方法: {spec_item.method}",
                filter_tags=current_filter_tags or [],
                terminated=True
            )
        except openai.APIError as e:
            print(f"  OpenAI API调用错误: {str(e)}。使用当前参数并终止。")
            return EvidenceSearchParams(
                query_text=current_query_text or f"{spec_item.content}\n评估方法: {spec_item.method}",
                filter_tags=current_filter_tags or [],
                terminated=True
            )
        except Exception as e:
            print(f"  生成搜索参数时发生意外错误: {str(e)}。使用当前参数并终止。")
            return EvidenceSearchParams(
                query_text=current_query_text or f"{spec_item.content}\n评估方法: {spec_item.method}",
                filter_tags=current_filter_tags or [],
                terminated=True
            )

    def generate_assessment(self, spec_item: AssessmentSpecItem, evidences: List[EvidenceSearchResult]) -> Conclusion:
        """
        使用 LLM 生成评估结论 (OpenAI 风格的 API 调用)。
        LLM 被要求输出采纳的证明材料的序号，函数随后解析这些序号以引用原始证明材料。

        Args:
            spec_item: 评估规范条目。
            evidences: NaviSearch 找到的相关证据片段列表。

        Returns:
            一个 Conclusion 对象，其中 evidence 字段包含对原始材料的引用。
        """
        print(f"调用 LLM，规范 ID {spec_item.id}，证据数量: {len(evidences)}...")

        # --- 构建 Prompt ---
        # 修改 Prompt，要求 LLM 返回选中证据的序号列表
        prompt_template = """
请根据以下网络安全评估规范条目和提供的证明材料，给出评估结论。
判断的取值应为以下三者之一：符合、不符合、不涉及。
并提供简要的补充说明。
同时，请从提供的证明材料列表中，选择并列出所有支持你结论的材料的序号（从1开始计数）。

评估规范条目ID: {spec_item_id}
评估规范标题: {spec_item_heading}
评估规范内容: {spec_item_content}
评估方法: {spec_item_method}

相关证明材料:
{formatted_evidences}

输出格式为JSON，包含以下字段:
"judgement": "符合/不符合/不涉及",
"comment": "补充说明...",
"selected_evidence_indices": [序号列表] // 例如: [1, 3] 表示选择了第1条和第3条材料

请确保 "selected_evidence_indices" 是一个包含整数序号的列表。如果没有任何材料支持你的结论，请返回一个空列表 []。
"""

        formatted_evidences = ""
        if evidences:
            for i, ev in enumerate(evidences):
                # 为每个证据编号，并在Prompt中展示部分内容以控制长度
                content_preview = ev.content
                if len(ev.content) > 500:  # 限制每个证据在Prompt中的预览长度
                    content_preview = ev.content[:500] + "..."
                formatted_evidences += f"材料 {i+1} (来源: {ev.source}):\n{content_preview}\n\n"
        else:
            formatted_evidences = "无相关证明材料。"

        prompt = prompt_template.format(
            spec_item_id=spec_item.id,
            spec_item_heading=spec_item.heading,
            spec_item_content=spec_item.content,
            spec_item_method=spec_item.method,
            formatted_evidences=formatted_evidences
        )
        # print("生成的 Prompt:", prompt) # 用于调试

        # 初始化一个默认的 Conclusion 对象，用于在出错时返回
        default_conclusion = Conclusion(
            judgement=Judgement.NOT_PROCESSED,
            comment="LLM调用或响应处理失败",
            evidence=[]
        )

        # --- 调用 OpenAI API 并解析响应 ---
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"} if "qwen" in self.model_name else None # 特定模型支持 JSON object format
            )

            llm_response_content = response.choices[0].message.content
            if not llm_response_content:
                default_conclusion.comment = "LLM返回了空响应。"
                return default_conclusion

            # 新增：去除Markdown代码块标记
            if llm_response_content.startswith('```json'):
                llm_response_content = llm_response_content[7:-3].strip()  # 去除```json和结尾的```
            llm_response = json.loads(llm_response_content)

            # 验证响应中是否包含所有必要字段
            required_keys = ["judgement", "comment", "selected_evidence_indices"]
            if not all(key in llm_response for key in required_keys):
                missing_keys = [key for key in required_keys if key not in llm_response]
                default_conclusion.comment = f"LLM响应缺少必要的字段: {', '.join(missing_keys)}。"
                return default_conclusion

            # 解析评估结论 (judgement)
            try:
                judgement_str = llm_response.get("judgement", "").strip().lower()
                if not judgement_str: # 处理空字符串的情况
                     raise ValueError("Judgement string is empty")
                judgement = Judgement(judgement_str)
            except ValueError:
                default_conclusion.comment = f"LLM返回了无效的评估结论: '{llm_response.get('judgement', '')}'。"
                # 尝试保留LLM的原始评论，如果它提供了关于为什么无法判断的线索
                if llm_response.get("comment"):
                    default_conclusion.comment += f" LLM原始评论: {llm_response.get('comment')}"
                return default_conclusion

            llm_comment = llm_response.get("comment", "") # 获取LLM的原始评论

            # 解析 selected_evidence_indices 并获取引用的原始证据
            selected_indices_from_llm = llm_response.get("selected_evidence_indices")
            referenced_evidences: List[EvidenceSearchResult] = []
            processing_comment = llm_comment # 初始化处理评论为LLM的原始评论

            if not isinstance(selected_indices_from_llm, list):
                processing_comment = f"LLM返回的 'selected_evidence_indices' 不是一个列表 (实际类型: {type(selected_indices_from_llm).__name__})。LLM原始评论: '{llm_comment}'"
            else:
                invalid_indices_found = False
                for index_val in selected_indices_from_llm:
                    if not isinstance(index_val, int):
                        processing_comment = f"LLM返回的 'selected_evidence_indices' 包含非整数值: '{index_val}'。LLM原始评论: '{llm_comment}'"
                        invalid_indices_found = True
                        break

                    actual_index = index_val - 1 # 将1-based转换为0-based
                    if 0 <= actual_index < len(evidences):
                        referenced_evidences.append(evidences[actual_index])
                    else:
                        processing_comment = f"LLM返回的证据序号 '{index_val}' 超出范围 (有效材料序号 1-{len(evidences)})。LLM原始评论: '{llm_comment}'"
                        invalid_indices_found = True
                        break

                if invalid_indices_found and judgement == Judgement.NOT_PROCESSED: # 如果因为索引错误导致无法判断
                    judgement = Judgement.ERROR # 可以设置一个特定的错误状态

            # 构建并返回最终的 Conclusion 对象
            return Conclusion(
                judgement=judgement,
                comment=processing_comment, # 使用处理后的评论
                evidence=referenced_evidences
            )

        except json.JSONDecodeError as e:
            default_conclusion.comment = f"无法解析LLM的JSON响应: {str(e)}。原始响应: '{llm_response_content if 'llm_response_content' in locals() else 'N/A'}'"
            return default_conclusion
        except openai.APIError as e: # Catching specific OpenAI API errors
            default_conclusion.comment = f"OpenAI API调用错误: {str(e)}"
            return default_conclusion
        except Exception as e:
            default_conclusion.comment = f"处理评估过程中发生意外错误: {str(e)}"
            return default_conclusion


if __name__ == "__main__":
    # 测试 LLMAssessmentClient 模块

    # 1. 创建模拟的 AssessmentSpecItem
    mock_spec_item = AssessmentSpecItem(
        id="TEST-001",
        condition="required",
        heading="测试评估项",
        content="这是一个用于测试的评估规范条目内容",
        method="文档审查和访谈"
    )

    # 2. 创建模拟的 EvidenceSearchResult 列表
    mock_evidences = [
        EvidenceSearchResult(
            source="测试文档1.pdf",
            content="在测试文档1中发现符合要求的配置记录，所有设置都按照标准执行。"
        ),
        EvidenceSearchResult(
            source="测试报告2.docx",
            content="访谈记录显示管理员了解相关安全要求，并已实施相应控制措施。"
        )
    ]

    # 3. 初始化 LLM 客户端
    # 注意：实际测试时需要设置环境变量或直接传入参数
    try:
        client = LLMAssessmentClient(
        )
        print("LLM 客户端初始化成功")
    except ValueError as e:
        print(f"初始化失败: {e}")
        exit(1)

    # 4. 测试正常情况下的评估生成
    print("\n测试1: 正常情况下的评估生成")
    result = client.generate_assessment(mock_spec_item, mock_evidences)
    print("评估结果:", result)

    # 5. 测试无证据的情况
    print("\n测试2: 无证据的情况")
    result = client.generate_assessment(mock_spec_item, [])
    print("评估结果:", result)

    # 6. 测试API错误情况（模拟无效URL）
    print("\n测试3: API错误情况")
    error_client = LLMAssessmentClient(
        llm_base_url="https://invalid-url.example.com",
        model_name="qwen-plus-latest"
    )
    result = error_client.generate_assessment(mock_spec_item, mock_evidences)
    print("错误处理结果:", result)

    # 7. 测试JSON解析错误情况（模拟无效响应）
    print("\n测试4: 模拟JSON解析错误")
    # 临时替换client的生成方法以模拟错误
    original_method = client.generate_assessment
    def mock_error_method(*args, **kwargs):
        return {"choices": [{"message": {"content": "这不是有效的JSON"}}]}
    client.generate_assessment = mock_error_method

    result = client.generate_assessment(mock_spec_item, mock_evidences)
    print("JSON解析错误结果:", result)

    # 恢复原始方法
    client.generate_assessment = original_method

    # 8. 测试验证错误情况（模拟无效结论）
    print("\n测试5: 模拟验证错误")
    def mock_validation_error(*args, **kwargs):
        return {"judgement": "无效选项", "comment": "测试评论"}
    client.generate_assessment = mock_validation_error

    result = client.generate_assessment(mock_spec_item, mock_evidences)
    print("验证错误结果:", result)

    # 恢复原始方法
    client.generate_assessment = original_method