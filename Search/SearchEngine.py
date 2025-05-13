from typing import List, Dict, Optional
from collections import Counter
from pymilvus import MilvusClient, Collection
from utils import get_embedding

class SearchEngine():
    def __init__(self, client:MilvusClient, collection_name:str):
        self.client = client
        self.collection_name = collection_name
        try:
            # 使用 pymilvus 的 Collection 类加载集合
            collection = Collection(name=self.collection_name)
            collection.load()
        except Exception as e:
            print(f"错误：加载集合失败: {e}")

    def retrieval(self, query_text:str, top_k:int = 5) -> List[Dict]:
        """
        根据自然语言查询文本，在Milvus中搜索最相似的评估规范。
        """
        try:
            query_embedding = get_embedding(query_text)
        except Exception as e:
            print(f"错误: 生成查询 embedding 失败: {e}")
            return [], []

        # 召回
        retrieval_param_dict = {
            "metric_type": "L2",
            "params": {"level": 1}  # 调整为你的索引参数
        }
        retrieval_records = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_param=retrieval_param_dict,
            limit=top_k,  # 初次召回更多样本
            output_fields=["id", "content", "tags"]
        )
        return [entity.get("entity") for entity in retrieval_records[0]]

    def rerank(self, filter_tags:List[str] = None, retrieval_records = None, mode:str = "ranking", top_k:int = 5):
        """_summary_

        Args:
            filter_tags (List[str], optional): _description_. Defaults to None.
            retrieval_records (_type_, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to "ranking".
            top_k (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        if filter_tags is None:
            filter_tags = []
        filter_set = set(filter_tags)

        # 根据模式处理记录
        if mode == "filtering":
            # 过滤模式：只保留命中所有filter_tags的文档
            if filter_tags == []:
                filtered_records = retrieval_records
            else:
                filtered_records = [
                    record for record in retrieval_records
                    if filter_set.issubset(set(record.get('tags', [])))
                ]
            # 按命中数量降序排序
            ranked_records = sorted(
                filtered_records,
                key=lambda d: len(set(d.get('tags', [])) & filter_set),
                reverse=True
            )
        elif mode == "ranking":
            # 排序模式：保留至少命中一个filter_tags的文档
            if filter_tags == []:
                filtered_records = retrieval_records
            else:
                filtered_records = [
                    record for record in retrieval_records
                    if filter_set and set(record.get('tags', [])) & filter_set
                ]
            # 按命中数量降序排序
            ranked_records = sorted(
                filtered_records,
                key=lambda d: len(set(d.get('tags', [])) & filter_set),
                reverse=True
            )
        else:
            ranked_records = retrieval_records  # 默认不处理
            raise ValueError(f"无效的模式: {mode}")


        ranked_records = ranked_records[:top_k]  # 取前top_k个文档

        # 统计top_k文档中的标签频数
        tags_counter = Counter()
        for doc in ranked_records:
            tags = doc.get('tags', [])
            unique_tags = set(tags)  # 每个文档的标签视为集合，避免重复计数
            tags_counter.update(unique_tags)

        # 计算阈值（top_k/2）
        threshold = top_k / 2
        # print(f"阈值: {threshold}")
        # print(f"标签出现次数: {tags_counter}")
        # 按abs(freq - threshold)降序排序标签
        ranked_tags = sorted(
            tags_counter.keys(),
            key=lambda tag: abs(tags_counter[tag] - threshold),
            reverse=True
        )
        print(f"排序后的标签: {ranked_tags}")
        return ranked_records, ranked_tags