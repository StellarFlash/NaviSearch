import os
import ast
import json
from typing import List, Tuple, Dict, Any, Union, Optional, TypeVar
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()  # 加载环境变量，包括API Key和base_url

embedding_base_url = os.getenv("EMBEDDING_BASE_URL","https://dashscope.aliyuncs.com/compatible-mode/v1")
embedding_api_key = os.getenv("EMBEDDING_API_KEY")

client = OpenAI(
    api_key=embedding_api_key,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url=embedding_base_url  # 百炼服务的base_url
)

embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))

def get_embedding(text:str, client:OpenAI = client, model:str = embedding_model, dim:int = embedding_dim) -> List[float]:
    """
    Args:
        text (str): _description_
        client (OpenAI, optional): _description_. Defaults to client.
        model (str, optional): _description_. Defaults to embedding_model.
        dim (int, optional): _description_. Defaults to embedding_dim.

    Returns:
        List[float]: _description_
    """
    print("调用embedding接口")
    completion = client.embeddings.create(
        model=model,
        input=text,
        dimensions=dim,
        encoding_format="float"
    )

    return completion.data[0].embedding

tagging_model = os.getenv("LLM_MODEL", "qwen-turbo")

def get_response(prompt:str = "", client:OpenAI = client, model:str = tagging_model) -> str:
    """
    Args:
        prompt (str, optional): _description_. Defaults to "".
        client (OpenAI, optional): _description_. Defaults to client.
        model (str, optional): _description_. Defaults to tagging_model.

    Returns:
        str: _description_
    """
    print("调用chat.completions接口")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业的助手，你会使用json字符串回复。"},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" }
    )
    return completion.choices[0].message.content.strip()

@staticmethod
def flatten_nested_structure(item: Any) -> List[str]:
    """
    递归地将嵌套结构扁平化为字符串列表，仅提取值。

    Args:
        item: 要处理的对象（str/int/float/bool/list/dict）

    Returns:
        扁平化字符串列表
    """
    flat_list = []

    if isinstance(item, str):
        try:
            parsed = json.loads(item)
            if isinstance(parsed, (dict, list)):
                return flatten_nested_structure(parsed)
        except json.JSONDecodeError:
            if item:
                flat_list.append(item)
    elif isinstance(item, (int, float, bool)):
        flat_list.append(str(item))
    elif isinstance(item, list):
        for element in item:
            flat_list.extend(flatten_nested_structure(element))
    elif isinstance(item, dict):
        for value in item.values():
            flat_list.extend(flatten_nested_structure(value))

    return flat_list

def tag_records(records: List, tags_design: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
    """
    为给定记录集合生成具有区分度的语义标签，并尽量与已有标签体系对齐。

    Args:
        records (List[Record]): 待打标的记录列表，每个记录应包含 id 和 content。
        tags_design (Dict[str, List[str]], optional): 已有的标签体系，用于引导新标签生成。默认为 None。

    Returns:
        Dict[str, List[str]]: 字典形式返回 ID → 标签列表 的映射关系。
    """
    if not records:
        return {}

    # 构建 Records 输入文本
    records_str = ""
    for record in records:
        records_str += f"""
        <record id="{record.id}">
        {record.content[:1000]}
        </record>
        """

    # 构建 Tags Design 输入文本
    tags_design_str = ""
    if tags_design:
        tags_design_str = ", ".join([f"{key}: {', '.join(val)}" for key, val in list(tags_design.items())[:10]])

    # 构建 Prompt
    prompt = f"""请为以下记录生成具有区分度的标签。规则如下：

1. 每条记录的输出是一个标签列表（多个标签之间用逗号分隔）；
2. 标签应当简洁、准确、可复用，使用行业术语或通用缩写（如 DDOS、SYNFlood）；
3. 尽量遵循已有的标签体系（如果提供），避免重复创建同类标签；
4. 每个标签控制在 1~5 个单词之间，优先首字母大写，无空格；
5. 最终输出必须是一个标准 JSON 对象，格式为 {{ "ID1": ["tagA", "tagB"], "ID2": [...] }}。

以下是你要处理的记录内容：
{records_str}

以下是参考标签体系（如有）：
{tags_design_str}

请直接输出 JSON 结果，不要添加任何解释：
"""

    # 调用 LLM
    try:
        response = get_response(prompt)
        tags = json.loads(response)

        # print("完成json解析")
        # 补全未返回标签的记录，确保所有记录都有标签
        result = {}
        for record in records:
            tags_int_keys = [int(k) for k in tags]
            if record.id in tags_int_keys:
                # print(f"[Info] Tags found for record {record.id}: {tags[str(record.id)]}")
                result[str(record.id)] = tags[str(record.id)]
            else:
                result[record.id] = []
                print(f"[Warning] No tags found for record {record.id}. Using empty list.")

        return result

    except Exception as e:
        print(f"[Error] Failed to parse LLM response: {e}")

        return {record.id: [] for record in records}


def tag_contents(contents: List[str], tags_design: Dict[str, List[str]] = None) -> List[List[str]]:
    """
    为给定的内容列表生成具有区分度的语义标签。输出为 {index: [tags]} 结构。
    仅对输出长度做检查，不维护 id 映射结构。

    参数:
        contents (List[str]): 输入的文本内容列表
        tags_design (Dict[str, List[str]]): 可选的标签体系参考

    返回:
        List[List[str]]: 每个内容对应的标签列表
    """
    # 构建 Records 输入文本（无 id）
    records_str = ""
    for index, content in enumerate(contents):
        records_str += f"""
        <content>
        {index}:
        {content[:2048]}
        </content>
        """

    # 构建 Tags Design 输入文本
    tags_design_str = ""
    if tags_design:
        tags_design_str = ", ".join([f"{key}: {', '.join(val)}" for key, val in list(tags_design.items())[:10]])

    # 构建 Prompt
    prompt = f"""请为以下内容生成具有区分度的语义标签。规则如下：
1. 标签应当简洁、准确、可复用，使用行业术语或通用缩写,如DDoS、网络攻击;
2. 你需要为每个内容生成一个标签列表，即使内容没有有意义的标签，也要使用空列表占位；
3. 尽量遵循已有的标签体系（如果提供），避免重复或冲突；
4. 每个标签控制在 1~5 个词，优先使用首字母大写形式；
5. 你输出的json字符串应当可以被解析为Dict[str, List[str]]，字典的键为序号，值为标签列表，使用双引号。
以下是要处理的内容：
{records_str}
以下为参考标签体系（如有）：
{tags_design_str}
请直接输出 JSON 结果，不要添加任何解释：
"""
    contents_tags_list = []
    # 调用 LLM
    try:
        response = get_response(prompt)
        # print(f"LLM响应{response}")
        response = response[response.find("{"):response.rfind("}")+1]

        contents_tags = ast.literal_eval(response)
        # print(f"解析结果{contents_tags}")
        for index, content in enumerate(contents):
            if str(index) in contents_tags.keys():
                # print(f"[Info] Tags found for content {index}: {contents_tags[str(index)]}")
                contents_tags_list.append(contents_tags[str(index)])
            else:
                contents_tags_list.append([])  # 无标签的内容使用空列表占位
        # print("完成更新")
        return contents_tags_list  # 返回解析后的 JSON 数据作为 respons
    except Exception as e:
        print(f"json解析:{response}失败")
        print(contents_tags)
        raise ValueError("[Error] Failed to parse LLM response")

filter_model = os.getenv("LLM_MODEL", "qwen-plus")
def get_filter(query_text:str, current_filter:List[str], recomanded_filter:List[str], current_iteration:int, current_size:int, max_iteration:int = 10, stop_size: int = 3) -> List[str]:
    """_summary_

    Args:
        query_text (str): _description_
        current_filter (List[str], optional): _description_. Defaults to None.

    Returns:
        List[str]: _description_
    """
    if current_filter is None:
        current_filter = []
    # 构建 prompt
    prompt = f"""
    你正在为用户执行文档搜索任务。你需要：
    1. 用户的查询文本是{query_text}，你需要找到最为相关的文档。
    2. 目前已经激活的过滤标签是{current_filter}。你需要从参考标签{recomanded_filter}中选择合适的标签，以进一步缩小候选文档的规模。
    3. 你需要将候选文档的规模缩小到{stop_size}内，目前候选文档的规模为{current_iteration}。
    4. 你总共拥有{max_iteration}次机会，目前已经使用{current_iteration}次，你应该谨慎地增加过滤条件，可以每次只增加一个过滤标签。
    5. 参考标签按照候选文档中出现的频数降序排列，靠前的标签有更高的召回率，但是特异性较低；靠后的标签特异性较高，但是召回率较低。
    6. 你应当直接回复json。键为filter，值为标签列表。
    """
    response = client.chat.completions.create(
        model=filter_model,
        messages=[
            {"role": "system", "content": "你是一个专业的语义标签设计师，你会使用json字符串回复。"},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" }
    ).choices[0].message.content.strip()

    try:
        response = response[response.find("{"):response.rfind("}")+1]
        response = json.loads(response)
        return response.get("filter", [])
    except Exception as e:
        print(f"json解析:{response}失败")
        return []

class Record:
    def __init__(self, id: str, content: str, tags: List[str]):
        self.id = id
        self.content = content
        self.tags = tags

def test_tag_records():
    records = [
        Record(id="R001", content="This is a DDoS attack using SYN Flood on IoT devices.", tags=[]),
        Record(id="R002", content="This malware encrypts files and asks for ransom.", tags=[]),
    ]
    tags_design = {
        "NetworkThreat": ["DDoS", "SYNFlood", "MITM"],
        "Malware": ["Ransomware", "Trojan", "Spyware"]
    }
    result = tag_records(records, tags_design)
    print(result)
    assert "R001" in result and "R002" in result

def test_tag_contents():
    contents = [
        "This is a DDoS attack using SYN Flood on IoT devices.",
        "This malware encrypts files and asks for ransom."

    ]
    contents_tags_list = tag_contents(contents)
    print(contents_tags_list)
    assert len(contents_tags_list) == len(contents)
if __name__ == "__main__":
    # test_tag_records()
    test_tag_contents()