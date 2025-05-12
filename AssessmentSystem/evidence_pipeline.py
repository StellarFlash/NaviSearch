import os
import json
import dotenv
import time
from typing import List, Dict, Any, Optional

# 导入第三方库和您自己的模块
from openai import OpenAI # 用于 MarkItDown 和 Metadata Tagger
from markitdown import MarkItDown # 用于文档转换
from langchain_openai import ChatOpenAI # 用于 Metadata Tagger
from langchain_core.documents import Document # Langchain 的 Document 模型
from langchain_community.embeddings import DashScopeEmbeddings # 用于生成 Embedding (可选，如果NaviSearch自动生成则不需要)
from langchain_community.document_loaders import DirectoryLoader, TextLoader # 用于加载文本文件
from langchain_community.document_transformers.openai_functions import create_metadata_tagger # 用于元数据标注
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter # 用于文档分割
from tqdm import tqdm # 显示处理进度

# 导入您自己定义的模型
from AssessmentSystem.model import EvidenceMaterial

# 假设您有一个用于处理文件转换的工具函数
# from RAG.PreProcess.utils import convert_and_save_files

dotenv.load_dotenv()

# 从环境变量加载配置
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 可能还需要其他 LLM 配置
LLM_MODEL_TRANSFORM = os.getenv("LLM_MODEL_TRANSFORM", "qwen-vl-ocr") # 用于格式转换的 LLM 模型
LLM_MODEL_TAGGING = os.getenv("LLM_MODEL_TAGGING", "qwen-turbo-latest") # 用于元数据标注的 LLM 模型
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3") # 用于生成 Embedding 的模型 (如果需要)

# 输入和输出目录配置
RAW_DOCS_DIR = "Docs/" # 存放原始文档的目录
MARKDOWN_DOCS_DIR = "Data/Markdown/" # 存放转换后 Markdown 文档的目录
TAGGED_DOCS_DIR = "Data/Tagged/" # 存放带元数据和分割后的文档片段 (中间结果)
FINAL_EVIDENCE_FILE = "AssessmentSystem/evidences.jsonl" # 最终生成的 JSONL 文件路径

# 元数据 Schema 文件路径
METADATA_SCHEMA_FILE = "Data/MetadataSchema/EvidenceMetadataSchema.json" # 请创建或修改此文件以定义元数据结构

class EvidenceProcessPipeline:
    """
    处理原始文档，生成可供 EvidenceLoader 加载的 JSONL 文件。
    流水线包括：格式转换 -> 加载 -> 分割 -> 元数据标注 -> 格式化输出。
    """
    def __init__(self):
        # 初始化客户端和工具
        self.llm_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        self.md_converter = MarkItDown(llm_client=self.llm_client, llm_model=LLM_MODEL_TRANSFORM)

        # 初始化 Langchain LLM 和 Embedding 模型 (用于标注和可能的 Embedding 生成)
        self.tagging_llm = ChatOpenAI(
            model=LLM_MODEL_TAGGING,
            base_url=DASHSCOPE_BASE_URL,
            api_key=DASHSCOPE_API_KEY,
            temperature=0.0,
        )

        # 初始化分割器
        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3'),
                ('####', 'Header 4'), ('#####', 'Header 5'), ('######', 'Header 6')
            ]
        )
        self.recursive_character_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", "", "<image_discription>", "</image_discription>"]
        )

        # 加载元数据 Schema
        self.metadata_schema = self._load_metadata_schema(METADATA_SCHEMA_FILE)
        # 初始化元数据标注器
        self.document_transformer = create_metadata_tagger(
            metadata_schema=self.metadata_schema,
            llm=self.tagging_llm
        )

        # 确保输出目录存在
        os.makedirs(MARKDOWN_DOCS_DIR, exist_ok=True)
        os.makedirs(TAGGED_DOCS_DIR, exist_ok=True)


    def _load_metadata_schema(self, file_path: str) -> Dict[str, Any]:
        """加载元数据 Schema 文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误：找不到元数据 Schema 文件: {file_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"错误：解析元数据 Schema 文件失败: {file_path}. 错误: {e}")
            raise
        except Exception as e:
            print(f"加载元数据 Schema 文件时发生意外错误: {e}")
            raise

    def transform_documents_to_markdown(self, input_dir: str, output_dir: str):
        """
        将指定目录下的原始文档转换为 Markdown 格式并保存。
        这里需要根据实际使用的 convert_and_save_files 函数实现。
        作为示例，我们假设它处理 input_dir 并输出到 output_dir。
        """
        print(f"开始将原始文档从 '{input_dir}' 转换为 Markdown 到 '{output_dir}'...")
        # 假设 convert_and_save_files 能够处理 input_dir 并将结果保存到 output_dir
        # 您需要根据您的 RAG.PreProcess.utils 模块实际实现来调用它
        # convert_and_save_files(self.md_converter, input_dir, output_dir)
        print("（此处应调用实际的文档转换函数）") # 占位符，需要您根据实际情况实现
        # 简单的占位符实现：遍历输入目录，复制文件（如果已经是md）或模拟转换过程
        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".md")
            if os.path.isfile(input_path):
                print(f"模拟处理文件: {filename}")
                # 模拟转换内容
                mock_content = f"# Document: {filename}\n\nContent of {filename} after conversion."
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(mock_content)
        print("文档转换模拟完成。")


    def load_and_split_markdown_docs(self, input_dir: str) -> List[Document]:
        """从 Markdown 文件加载文档并进行分割。"""
        print(f"开始加载并分割 Markdown 文档从 '{input_dir}'...")
        loader = DirectoryLoader(input_dir, glob="**/*.md", show_progress=True, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()
        print(f"加载了 {len(docs)} 个 Markdown 文件。")

        doc_splits = []
        print("开始分割文档...")
        for doc in tqdm(docs, desc="分割文档"):
            # 将原始文档按 Markdown 头部分割
            header_splits = self.markdown_header_splitter.split_text(doc.page_content)
            for split in header_splits:
                 # 对于长度过长或过短的片段进行进一步处理
                 if len(split.page_content) < 10: # 忽略过短片段
                     continue
                 elif len(split.page_content) > 5000: # 对超长片段进行递归分割
                     recursive_splits = self.recursive_character_splitter.split_documents([split])
                     doc_splits.extend(recursive_splits)
                 else:
                     doc_splits.append(split)

        print(f"分割后得到 {len(doc_splits)} 个文档片段。")
        return doc_splits

    def tag_document_metadata(self, doc_splits: List[Document]) -> List[Document]:
        """使用 LLM 为文档片段标注元数据。"""
        print("开始标注文档片段元数据...")
        # 使用 tqdm 显示进度
        tagged_splits = []
        with tqdm(total=len(doc_splits), desc='标注元数据') as pbar:
             # Langchain 的 transform_documents 函数本身通常会处理迭代
             tagged_splits = self.document_transformer.transform_documents(doc_splits)
             pbar.update(len(doc_splits)) # tqdm 需要手动更新进度

        print("元数据标注完成。")
        return tagged_splits

    def format_to_evidence_jsonl(self, tagged_splits: List[Document], output_file: str,
                                 project: Optional[str] = None, collector: Optional[str] = None,
                                 collection_time: Optional[int] = None):
        """
        将带元数据的文档片段格式化为 EvidenceMaterial 并保存为 JSONL 文件。
        Args:
            tagged_splits: 带有元数据的文档片段列表。
            output_file: 输出的 JSONL 文件路径。
            project: 指定的所属项目名称。
            collector: 指定的采集人名称。
            collection_time: 指定的采集时间戳。
        """
        print(f"开始格式化并保存证据到 '{output_file}'...")
        evidences: List[EvidenceMaterial] = []
        for split in tagged_splits:
            try:
                # 获取文件路径元数据，用于提取文件名作为 source
                file_path = split.metadata.get("file_path")
                source_filename = os.path.basename(file_path) if file_path else "unknown"

                evidence_data = {
                    "content": split.page_content,
                    # 优先使用标注器提供的 tags，如果没有则为空字典
                    "tags": split.metadata.get("tags", {}),
                    # 使用从 file_path 中提取的文件名作为 source
                    "source": source_filename,
                    # 使用运行时指定的 project 参数
                    "project": project,
                    # 使用运行时指定的 collection_time 参数
                    "collection_time": collection_time,
                    # 使用运行时指定的 collector 参数
                    "collector": collector
                    # "embedding": self.embeddings.embed_query(split.page_content) # 如果需要在此生成 Embedding
                    # 其他字段根据元数据 Schema 映射
                }
                evidences.append(EvidenceMaterial(**evidence_data))
            except Exception as e:
                # 增强错误信息，包含文件名和片段内容
                source_info = split.metadata.get("file_path", "未知文件")
                print(f"警告：格式化文档片段时出错，跳过。来自文件: {source_info}，片段内容开头：{split.page_content[:100]}... 错误: {e}")
                continue # 跳过格式化失败的片段


        # 写入 JSONL 文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for evidence in tqdm(evidences, desc="写入 JSONL"):
                    # 使用 model_dump_json 方法将 Pydantic 模型转换为 JSON 字符串
                    f.write(evidence.model_dump_json(ensure_ascii=False) + '\n')
            print(f"成功将 {len(evidences)} 条证据保存到 {output_file}")
        except Exception as e:
            print(f"写入 JSONL 文件时发生错误: {e}")
            raise e

    def run(self, project: Optional[str] = None, collector: Optional[str] = None, collection_time: Optional[int] = None):
        """
        运行整个文档处理流水线。
        Args:
            project: 指定的所属项目名称。
            collector: 指定的采集人名称。
            collection_time: 指定的采集时间戳 (默认为当前处理时间)。
        """
        print("开始执行证据处理流水线...")

        # 如果未指定采集时间，则使用当前时间
        if collection_time is None:
            collection_time = int(time.time())
            print(f"未指定采集时间，使用当前时间戳: {collection_time}")
        else:
             print(f"使用指定的采集时间戳: {collection_time}")

        # 步骤 1: 文档格式转换 (需要您实现 convert_and_save_files 或替换此步骤)
        # self.transform_documents_to_markdown(RAW_DOCS_DIR, MARKDOWN_DOCS_DIR)
        print("跳过实际文档格式转换步骤，请确保 Data/Markdown/ 目录中已有 .md 文件用于测试。")
        # 为了演示，跳过实际转换，假设 Markdown 文件已存在于 MARKDOWN_DOCS_DIR

        # 步骤 2&3: 加载 Markdown 文档并进行分割
        markdown_splits = self.load_and_split_markdown_docs(MARKDOWN_DOCS_DIR)

        if not markdown_splits:
            print("没有找到可处理的 Markdown 文档片段，流水线结束。")
            return

        # 4. 元数据提取和标注
        # 注意：这里的标注可能会尝试提取 project 和 source，但我们将在后续步骤中覆盖它们以使用运行时指定的值
        tagged_splits = self.tag_document_metadata(markdown_splits)

        if not tagged_splits:
            print("元数据标注后没有得到有效的文档片段，流水线结束。")
            return

        # 5. 格式化为 EvidenceMaterial 并保存为 JSONL
        # 将运行时指定的 project, collector 和 collection_time 传递下去
        self.format_to_evidence_jsonl(
            tagged_splits,
            FINAL_EVIDENCE_FILE,
            project=project,
            collector=collector,
            collection_time=collection_time
        )

        print("证据处理流水线执行完毕。")


# 示例用法
if __name__ == "__main__":
    # 确保您的 .env 文件中设置了 DASHSCOPE_BASE_URL 和 DASHSCOPE_API_KEY

    # ... (创建示例 Schema 和 Markdown 文件的代码保持不变) ...

    pipeline = EvidenceProcessPipeline()

    # 在运行时指定参数
    my_project = "网络安全评估项目"
    my_collector = "张三" # 或者从某个配置或命令行参数获取
    # my_collection_time = int(time.time()) # 如果需要指定一个固定的时间，比如任务开始时间
    # 如果不指定 collection_time，run 方法将使用当前时间

    print(f"准备运行流水线，指定项目: '{my_project}', 采集人: '{my_collector}'")

    # 运行流水线，并传入指定的参数
    pipeline.run(project=my_project, collector=my_collector)
    # 如果需要指定采集时间，可以这样调用：
    # pipeline.run(project=my_project, collector=my_collector, collection_time=my_collection_time)