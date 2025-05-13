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
    def __init__(self, llm_base_url: str = None, api_key: str = None, model_name: str = "qwen-plus-latest", temperature: float = 0.0):

        # 如果没有显式提供，则从环境变量加载
        self.base_url = llm_base_url or os.getenv("LLM_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model_name = model_name or os.getenv("LLM_MODEL")
        self.temperature = temperature # Add temperature as an initialization parameter

        if not self.base_url:
            raise ValueError("LLM_BASE_URL 必须设置，可以作为参数或在 .env 文件中设置")
        if not self.api_key:
            print("警告：没有 API 密钥，LLM 服务可能无法工作！") # Optional: Warning if no key provided
        if not self.model_name:
           raise ValueError("LLM_MODEL 必须设置，可以作为参数或在 .env 文件中设置")

        print(f"LLMAssessmentClient 初始化，API URL: {self.base_url}, 模型: {self.model_name}")

        self.client = OpenAI(base_url = self.base_url,api_key = self.api_key) # Initialize OpenAI client with base_url and api_key (if provided)are headers for API calls (if using an API key)

    def generate_search_params(self, spec_item: AssessmentSpecItem) -> EvidenceSearchParams:
        """
        生成搜索参数，根据评估规范内容生成查询文本。
        Args:
            spec_item: 评估规范条目。
        Returns:
            一个包含查询文本的 EvidenceSearchParams 对象。
        """
        return EvidenceSearchParams(query_text=spec_item.content + "\n" + spec_item.method + "\n", tags = [])  # 简单地使用 spec_item 的内容作为查询文本

    def generate_assessment(self, spec_item: AssessmentSpecItem, evidences: List[EvidenceSearchResult]) -> Dict:
        """
        使用 LLM 生成评估结论 (OpenAI 风格的 API 调用).
        Args:
            spec_item: 评估规范条目。
            evidences: NaviSearch 找到的相关证据片段列表。
        Returns:
            一个包含 'judgement' 和 'comment' 键的字典。
        """
        print(f"调用 LLM，规范 ID {spec_item.id}，证据数量: {len(evidences)}...")
        # --- 构建 Prompt（类似于设计说明） ---
        prompt_template = """
请根据以下网络安全评估规范条目和提供的证明材料，给出评估结论，并附上支持结论的材料来源。
判断的取值应为以下三者之一：符合、不符合、不涉及。
并提供简要的补充说明。
评估规范条目ID: {spec_item_id}
评估规范标题: {spec_item_heading}
评估规范内容: {spec_item_content}
评估方法: {spec_item_method}
相关证明材料:
{formatted_evidences}
输出格式为JSON:
"judgement": "符合/不符合/不涉及",
"comment": "补充说明...",
"evidence": [
    "source":"材料1",
    "content":"与评估直接相关的证明材料的段落，使用原始表达，保留完整段落"

    "source":"材料2",
    "content":"与评估直接相关的证明材料的段落"
]

"""
        # print("prompt_template:", prompt_template)  # 打印完整的 Prompt，用于调试
        formatted_evidences = ""
        if evidences:
            for i, ev in enumerate(evidences):
                formatted_evidences += f"材料 {i+1} (来源: {ev.source}):\n{ev.content[:200]}...\n\n" # 限制 prompt 中的证据内容长度
        else:
            formatted_evidences = "无相关证明材料。"
        prompt = prompt_template.format(
            spec_item_id=spec_item.id,
            spec_item_heading=spec_item.heading,
            spec_item_content=spec_item.content,
            spec_item_method=spec_item.method,
            formatted_evidences=formatted_evidences
        )
        # print("生成的 Prompt:", prompt)  # 打印完整的 Prompt，用于调试
        conclusion = Conclusion(
            spec_item_id=spec_item.id,
            judgement=Judgement.NOT_PROCESSED,  # 默认值
            comment="LLM返回的响应格式不正确",
            evidence=[]
        )
        # --- 解析 OpenAI API 响应 ---
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"} if self.model_name == "qwen-plus-latest" else None
            )

            # 解析响应
            # 解析响应并验证
            llm_response = json.loads(response.choices[0].message.content)

            if not all(key in llm_response for key in ["judgement", "comment", "evidence"]):
                conclusion.comment = "LLM响应缺少必要的字段"
                return conclusion

            # 将LLM返回的judgement字符串转换为Judgement枚举
            try:
                judgement = Judgement(llm_response["judgement"].lower())
            except ValueError:
                conclusion.comment = f"无效的评估结论: {llm_response['judgement']}"
                return conclusion

            # 构建并返回Conclusion对象
            return Conclusion(
                spec_item_id=spec_item.id,
                judgement=judgement,
                comment=llm_response["comment"],
                evidence=llm_response["evidence"]
            )

        except json.JSONDecodeError as e:
            conclusion.comment = f"无法解析LLM的JSON响应: {str(e)}"
            return conclusion
        except openai.APIError as e:
            conclusion.comment = f"OpenAI API错误: {str(e)}"
            return conclusion
        except Exception as e:
            conclusion.comment = f"发生意外错误: {str(e)}"
            return conclusion


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