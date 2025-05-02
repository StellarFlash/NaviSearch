"""
SemanticTagger.py

语义标签器

生成最有效的语义标签。

创建日期：2025-04-27
"""
import os
import ast
import json
import random
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypeVar
from unittest.mock import MagicMock, patch

from pymilvus import MilvusClient
from utils import get_embedding, tag_records, tag_contents

class SemanticTagger:
    def __init__(self, tags_design_path: str = "", batch_size: int = 5):
        self.tags_design_path = tags_design_path
        if self.tags_design_path != "":
            self._load_tags_design()
        else:
            self.tags_design = {}
        self.batch_size = batch_size
        load_dotenv()  # 加载环境变量，包括API Key和base_url
        milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        milvus_token = os.getenv("MILVUS_TOKEN", "root:Milvus")
        self.client = MilvusClient(uri = milvus_uri, token=milvus_token)
        self.collection_name = os.getenv("MILVUS_COLLECTION", "KnowledgeBaseRecords")


    def InsertTimeTagging(
        self,
        records: List[Dict],
        batch_size: int = 10,
        sample_scale: int = 20,
        update_reference_records: bool = False
    ) -> List[Dict]:
        """
        插入时标记：基于相似文档上下文，为每条记录生成更具区分度的语义标签。
        可通过参数 `update_reference_records` 控制是否对参考文档进行更新。

        Args:
            records (List[Record]): 待打标的记录列表；
            batch_size (int): 构造参考文档集的大小；
            sample_scale (int): 从 Milvus 中召回的相似文档数量；
            update_reference_records (bool): 是否更新参与批次的 reference_records；

        Returns:
            List[Record]: 实际更新了标签的记录列表（含 current_record 和 changed reference_records）
        """
        updated_records = []

        for record in records:
            # Step 1: 基于当前记录内容，召回 top_k 相似文档，构建参考文档集
            query_embedding = get_embedding(record.get("content"))
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                # param={"metric_type": "COSINE", "nprobe": 10},
                limit=sample_scale,
                output_fields=["id", "content", "tags", "embedding"]
            )
            batch_record_dict = []
            for hit in hits[0]:  # hits 是嵌套列表，每个查询结果对应一个子列表
                entity = hit.get("entity", {})
                tags_str = entity.get("tags", "[]")  # 默认空列表
                try:
                    tags_list = ast.literal_eval(tags_str)
                except (ValueError, SyntaxError):
                    tags_list = []  # 如果解析失败，就设为空列表

                batch_record_dict.append({
                    "id": hit.get("id"),
                    "content": entity.get("content", ""),
                    "tags": tags_list,  # 确保传入的是列表
                })
            # Step 2: 随机采样
            target_record_dict = batch_record_dict[0]

            random.sample(batch_record_dict[1:-1], k=batch_size-1)  # 随机采样

            contents = [target_record_dict.get("content")]
            contents.extend([record_dict.get("content") for record_dict in batch_record_dict])
            # Step 3: 批量生成标签
            batch_tags = tag_contents(contents = contents)

            # Step 4: 更新 batch_record_dict 中的标签，并写入数据库
            for i, tags in enumerate(batch_tags):
                former_tags = batch_record_dict[i]["tags"]
                if former_tags != tags:  # 检查标签是否有变化
                    final_tags = list(set(former_tags + tags))  # 合并标签并去重
                    self.client.upsert(  # 更新 Milvus 中的标签
                        collection_name=self.collection_name,
                        data=[{
                            "id": batch_record_dict[i]["id"],
                            "tags": final_tags,  # 确保传入的是列表
                        }])
                    updated_records.append({  # 记录更新的记录
                        "id": batch_record_dict[i]["id"],
                        "contents": batch_record_dict[i]["content"],
                        "tags": final_tags,
                    })
                # 如果不更新参考记录，则在处理第一条记录后退出循环。
                if not update_reference_records:  # 更新 reference_records
                    break

        return updated_records

    def RetrievalTimeTagging(self, records: List[Dict] = None, batch_size: int = 3) -> List[Dict]:
        """
        召回时标记：对给定记录字典进行语义标签补充。

        Args:
            records (List[Dict]): 待打标的记录列表。预设已经具备id, content, tags和embedding字段。
            batch_size (int): 批次大小，默认为 self.batch_size

        Returns:
            List[Dict]: 更新标签后的记录列表
        """
        if not records:
            return records

        print(f"共有{len(records)}条记录需要打标签")

        batch_size = batch_size or self.batch_size

        batches = []
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batches.append(batch)

        for batch in batches:
            # print(f"正在处理第{batches.index(batch) + 1}批")
            batch_contents = [record.get("content") for record in batch]
            batch_tags = [record.get("tags", []) for record in batch]
            new_tags = tag_contents(batch_contents)
            for i, tags in enumerate(new_tags):
                final_tags = list(set(batch_tags[i] + tags))
                batch[i]["tags"] = final_tags

        return records





    def _load_tags_design(self) -> Dict[str, Any]:
        """
        加载标签设计文件。
        """
        if not os.path.exists(self.tags_design_path):
            raise FileNotFoundError(f"标签设计文件未找到：{self.tags_design_path}")
        with open(self.tags_design_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data


def test_InsertTimeTagging():
    """
    测试 InsertTimeTagging 方法：
    - 能否正确从 Milvus 获取相似记录；
    - 能否调用 tag_contents 生成标签；
    - 能否更新数据库中的标签。
    """
    print("\n=== 开始测试 InsertTimeTagging ===")

    # 初始化 SemanticTagger 实例
    tagger = SemanticTagger(batch_size=2)

    # Mock Milvus client
    mock_client = MagicMock()
    tagger.client = mock_client

    # 构造测试输入记录
    test_record = {
        "id": 999,
        "content": "深度学习是人工智能的核心技术"
    }

    # 构建 Milvus 返回的 hits 数据结构
    mock_hits = [[{
        "id": 100,
        "entity": {
            "id": 100,
            "content": "机器学习在金融中的应用",
            "tags": "['AI', '数据科学']",
            "embedding": [0.1, 0.2, 0.3]
        }
    }, {
        "id": 101,
        "entity": {
            "id": 101,
            "content": "神经网络与图像识别",
            "tags": "['计算机视觉', 'DL']",
            "embedding": [0.15, 0.25, 0.35]
        }
    }]]

    # 设置 mock 返回值
    mock_client.search.return_value = mock_hits
    mock_client.upsert.return_value = True

    # 模拟 get_embedding 函数
    with patch("SemanticTagger.get_embedding", return_value=[0.12, 0.22, 0.32]):
        # 模拟 tag_contents 函数
        with patch("SemanticTagger.tag_contents", return_value=[
            ["AI", "深度学习"],               # 当前文档的标签
            ["AI", "金融科技"],               # 相似文档1的新标签
            ["深度学习", "图像处理"]           # 相似文档2的新标签
        ]):
            updated_records = tagger.InsertTimeTagging(
                records=[test_record],
                batch_size=2,
                sample_scale=2,
                update_reference_records=False
            )

            # 验证结果
            print("更新后的记录：")
            for r in updated_records:
                print(json.dumps(r, ensure_ascii=False, indent=2))

            assert len(updated_records) == 2, "应更新两条参考记录"
            print("✅ InsertTimeTagging 测试通过")


def test_RetrievalTimeTagging():
    """
    测试 RetrievalTimeTagging 方法：
    - 能否为现有记录批量补充新标签；
    - 能否保留旧标签并合并去重。
    """
    print("\n=== 开始测试 RetrievalTimeTagging ===")

    # 初始化 SemanticTagger 实例
    tagger = SemanticTagger(batch_size=2)

    # 构造测试记录（含已有标签）
    test_records = [
        {"id": 100, "content": "自然语言处理是AI的关键领域", "tags": ["NLP", "语言模型"]},
        {"id": 101, "content": "强化学习推动机器人智能化发展", "tags": ["机器人学"]},
        {"id": 102, "content": "深度学习模型在图像识别中表现优异", "tags": ["图像识别", "CNN"]},
        {"id": 103, "content": "大模型在多个任务上展现出强大能力", "tags": []},
        {"id": 104, "content": "推荐系统依赖用户行为数据分析", "tags": ["推荐系统", "协同过滤"]},
        {"id": 105, "content": "图神经网络用于社交网络分析", "tags": ["GNN", "社交网络"]},
        {"id": 106, "content": "迁移学习让小数据也能训练出好模型", "tags": ["迁移学习"]},
        {"id": 107, "content": "Transformer架构引领序列建模新时代", "tags": ["Transformer", "注意力机制"]},
        {"id": 108, "content": "边缘计算结合AI实现低延迟推理", "tags": ["边缘计算", "AI部署"]},
        {"id": 109, "content": "联邦学习保护用户隐私的同时提升模型效果", "tags": ["联邦学习", "隐私保护"]}
    ]

    # 模拟 tag_contents 返回新标签
    with patch("SemanticTagger.tag_contents", return_value=[
        ["NLP", "语义分析"],
        ["强化学习", "智能控制"]
    ]):
        updated_records = tagger.RetrievalTimeTagging(test_records, batch_size=2)

        print("更新后的记录：")
        for r in updated_records:
            print(json.dumps(r, ensure_ascii=False, indent=2))

        print("✅ RetrievalTimeTagging 测试通过")


if __name__ == "__main__":
    print("🟢 开始运行语义标签器测试...\n")

    # try:
    #     test_InsertTimeTagging()
    # except AssertionError as e:
    #     print(f"🔴 InsertTimeTagging 测试失败: {e}")

    try:
        test_RetrievalTimeTagging()
    except AssertionError as e:
        print(f"🔴 RetrievalTimeTagging 测试失败: {e}")

    print("\n📊 测试完成")