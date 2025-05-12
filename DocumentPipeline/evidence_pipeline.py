import os
import json
import dotenv
import time

import re # 导入用于处理正则表达式的re模块
import glob # 导入用于查找文件的glob模块
import pypandoc # 导入用于处理文档转换的pypandoc模块
from pathlib import Path # 导入用于路径操作的Path模块
from datetime import datetime
from typing import List, Dict, Any, Optional

# 导入第三方库和您自己的模块
from openai import OpenAI # 用于 MarkItDown 和 Metadata Tagger
from markitdown import MarkItDown # 用于文档转换和图片理解
from langchain_openai import ChatOpenAI # 用于 Metadata Tagger
from langchain_core.documents import Document # Langchain 的 Document 模型
# from langchain_community.embeddings import DashScopeEmbeddings # 用于生成 Embedding (可选)
from langchain_community.document_loaders import DirectoryLoader, TextLoader # 用于加载文本文件
from langchain_community.document_transformers.openai_functions import create_metadata_tagger # 用于元数据标注
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter # 用于文档分割
from tqdm import tqdm # 显示处理进度

# 导入您自己定义的模型 (请确保 AssessmentSystem.model 存在并定义了 EvidenceMaterial)
try:
    from AssessmentSystem.model import EvidenceMaterial
except ImportError:
    print("警告: 未找到 AssessmentSystem.model 或 EvidenceMaterial 定义。将使用一个简单的字典结构代替。")
    # 如果找不到模型，定义一个简单的字典结构作为替代
    class EvidenceMaterial:
        def __init__(self, content, tags, source, project, collector, collection_time, embedding=None):
            self.content = content
            self.tags = tags
            self.source = source
            self.project = project
            self.collector = collector
            self.collection_time = collection_time
            self.embedding = embedding

        def model_dump_json(self, ensure_ascii=False):
            """模拟 Pydantic 的 model_dump_json 方法"""
            data = {
                "content": self.content,
                "tags": self.tags,
                "source": self.source,
                "project": self.project,
                "collector": self.collector,
                "collection_time": self.collection_time,
                "embedding": self.embedding # 注意：此处 embedding 字段并未在此流水线中生成
            }
            return json.dumps(data, ensure_ascii=ensure_ascii)


dotenv.load_dotenv()

# 从环境变量加载配置
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 可能还需要其他 LLM 配置
LLM_MODEL_TRANSFORM = os.getenv("LLM_MODEL_TRANSFORM", "qwen-vl-ocr") # 用于格式转换和图片理解的 LLM 模型
LLM_MODEL_TAGGING = os.getenv("LLM_MODEL_TAGGING", "qwen-turbo-latest") # 用于元数据标注的 LLM 模型
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3") # 用于生成 Embedding 的模型 (如果需要)

# 输入和输出目录配置
RAW_DOCS_DIR = "Docs/" # 存放原始文档的目录 (如 .docx, .pdf)
MARKDOWN_DOCS_DIR = "Data/Markdown/" # 存放转换后 Markdown 文档的目录
# TAGGED_DOCS_DIR = "Data/Tagged/" # 存放带元数据和分割后的文档片段 (中间结果，当前代码不直接使用此目录)
FINAL_EVIDENCE_FILE = "AssessmentSystem/evidences.jsonl" # 最终生成的 JSONL 文件路径

# 元数据 Schema 文件路径
METADATA_SCHEMA_FILE = "DocumentPipeline/evidence_schema.json" # 请确保此文件存在并定义了元数据结构


# --- 工具函数，直接在此模块内实现 ---

def format_ocr_result(ocr_result: str) -> str:
    """
    格式化OCR识别结果，将其转换为Markdown格式，并包裹在特殊标签中。
    """
    formatted_ocr_result = ""
    # 假设OCR结果是按行返回的文本
    for line in ocr_result.split('\n'):
        if line.strip():  # 确保不是空行或只有空白字符的行
            # 将每一行作为Markdown的一个段落或列表项，这里简单处理为引用块内段落
            formatted_ocr_result += f"> {line.strip()}\n" # strip() 去除行首尾空白

    # 使用特殊标签包裹图片描述
    formatted_ocr_result = "<image_discription>\n" + formatted_ocr_result + "</image_discription>\n"
    return formatted_ocr_result


def process_markdown_images(md_converter: MarkItDown, markdown_str: str, base_dir: str) -> str:
    """
    处理Markdown字符串中的图片引用，使用md_converter（视觉LLM）获取描述文本，
    并将其添加到图片引用下方。使用 pathlib 健壮处理图片路径。

    参数:
        md_converter: 转换器对象，需要有 convert(media_file_path) 方法，用于处理图片。
        markdown_str: 要处理的Markdown字符串。
        base_dir: Markdown 文件所在的基准目录，用于构建图片的绝对路径。
                  这个目录应该是 Pandoc 转换后 Markdown 文件所在的目录。

    返回:
        处理后的Markdown字符串。
    """
    # 正则表达式匹配图片引用块及其可能的多行引用前缀 (>)
    # 匹配格式: > ![](path){width="x" height="y"} 可能多行连续
    # 改进的正则表达式，更健壮地匹配图片引用及其之前的可选引用符号和空白
    pattern = re.compile(
        r'(^>\s*!\[.*?\]\((.+?)\).*?)(?=\n(?!>\s*!\[|\s*$)|$)', # 匹配以 > 开头（可选空白）的图片引用，直到下一行不是引用或文件结束
        re.MULTILINE | re.DOTALL # 允许多行匹配和点号匹配换行符
    )

    # 替换函数
    def replace_match(match):
        full_match = match.group(1)  # 完整匹配内容
        relative_media_path_str = match.group(2)  # 图片路径字符串，可能包含相对路径或奇怪的格式

        # 构建图片的绝对路径 using pathlib
        media_path = None
        try:
            # 使用 Path 对象处理路径拼接
            # Path(base_dir) 是 Markdown 文件所在的目录
            # relative_media_path_str 是从 Markdown 中提取的图片路径
            # / 操作符会根据操作系统正确拼接路径
            media_path_obj = relative_media_path_str

            # 使用 .resolve() 来解析最终的绝对路径
            # strict=True 会检查路径是否存在，不存在则抛出 FileNotFoundError
            # strict=False 不检查是否存在，只解析路径
            # 考虑到图片可能在处理过程中被移动或删除，或者路径解析有误，这里先用 strict=False
            # 在实际处理图片时再捕获 FileNotFoundError
            media_path = media_path_obj

            print(f"--- Debug Image Path ---")
            print(f"base_dir: {base_dir}")
            print(f"relative_media_path (captured): {relative_media_path_str}")
            print(f"Constructed media_path: {media_path}") # 检查这个路径是否正确


        except Exception as e:
             print(f"错误: 构建图片路径失败 {relative_media_path_str} 相对 {base_dir}. 错误: {e}")
             # 返回原始匹配内容，跳过此图片处理
             return full_match


        try:
            # 调用转换器获取描述 (using the constructed media_path)
            # 在这里使用 strict=True 尝试打开文件，如果文件不存在会抛出 FileNotFoundError
            final_media_path = str(Path(media_path).resolve(strict=True)) # 再次尝试解析并验证路径是否存在
            print(f"正在处理图片: {final_media_path}")  # 调试信息

            # md_converter.convert 应该能处理图片文件路径
            result = md_converter.convert(final_media_path)
            # 假设 convert 返回的结果有一个 .markdown 属性包含OCR/描述文本
            description = format_ocr_result(result.markdown if hasattr(result, 'markdown') else str(result))
            # 限制打印的描述长度，避免控制台输出过多
            print(f"转换结果: {description[:100].replace('\n', ' ') + '...'}") # 调试信息，打印部分描述，替换换行符方便查看

            # 在原始内容后添加描述
            return f"{full_match}\n{description}"

        except FileNotFoundError:
             print(f"警告: 图片文件未找到，跳过处理: {media_media_path}. 请检查文件是否存在。")
             return full_match # 出错时返回原内容
        except Exception as e:
            print(f"处理图片 {media_path} 时出错: {str(e)}")
            return full_match  # 出错时返回原内容

    # 执行替换并返回结果
    # base_dir is passed from convert_and_save_files, which is the directory of the .md file
    processed_markdown = pattern.sub(replace_match, markdown_str)
    return processed_markdown


def convert_and_save_files(
    md_converter: MarkItDown,
    source_dir: str,
    target_dir: str,
    file_patterns: List[str] = ["*.docx", "*.pdf"], # 添加 .pdf 支持，如果 MarkItDown 支持
    overwrite: bool = False
) -> None:
    """
    遍历源目录中符合通配符的文件，转换为Markdown并保存到目标目录，保持原始目录结构。
    对于 .docx 文件，使用 pypandoc 进行转换并提取媒体资源，再用md_converter处理图片。
    对于其他文件，使用传入的 md_converter 进行转换并处理图片（如果转换器支持）。

    Args:
        md_converter: 具有convert方法的转换器对象 (用于非 .docx 文件和图片处理)。
                      需要支持处理文件路径和图片路径。
        source_dir: 源目录路径。
        target_dir: 目标目录路径。
        file_patterns: 要匹配的文件通配符列表。
        overwrite: 是否覆盖已存在的目标文件，默认为False。
    """
    # 确保目标目录存在
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # 遍历每种文件模式
    for pattern in file_patterns:
        # 使用glob递归查找所有匹配的文件
        # 使用 os.path.join 构建跨平台的路径
        search_path = os.path.join(source_dir, "**", pattern)
        print(f"正在搜索文件: {search_path}")
        # glob.glob 返回的路径是相对于当前工作目录的
        for src_file_path in glob.glob(search_path, recursive=True):
            print(f"发现文件: {src_file_path}")
            # 获取相对于源目录的相对路径，以便在目标目录中重建结构
            # 使用 Path().relative_to() 更准确地获取相对路径
            src_file_path_obj = Path(src_file_path)
            source_dir_obj = Path(source_dir)
            try:
                 relative_path_obj = src_file_path_obj.relative_to(source_dir_obj)
                 relative_path = str(relative_path_obj)
            except ValueError:
                 # 如果文件不在 source_dir 内 (不应该发生，除非 glob 有问题)
                 print(f"警告: 文件 {src_file_path} 不在源目录 {source_dir} 内，跳过。")
                 continue


            # 构造目标文件路径（将扩展名改为.md）
            # 使用 Path().with_suffix('.md') 更改扩展名
            target_file_path_obj = Path(target_dir) / relative_path_obj.with_suffix('.md')
            target_file_path = str(target_file_path_obj)

            # 如果文件已存在且不覆盖，则跳过
            if not overwrite and os.path.exists(target_file_path):
                print(f"跳过已存在的文件: {target_file_path}")
                continue

            # 确保目标文件所在的目录存在
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

            markdown_content = None # 用于存储转换后的 Markdown 内容

            try:
                # 获取文件扩展名
                file_extension = os.path.splitext(src_file_path)[1].lower()

                if file_extension == ".docx":
                    # --- 处理 .docx 文件使用 pypandoc ---
                    print(f"正在使用 pypandoc 转换: {src_file_path} -> {target_file_path}")

                    # 构建媒体资源保存目录路径，相对于目标 Markdown 文件所在的目录
                    # 例如：target_dir/path/to/filename_media
                    media_dir_name = os.path.splitext(os.path.basename(src_file_path))[0] + "_media"
                    media_dir_path = os.path.join(os.path.dirname(target_file_path), media_dir_name)
                    os.makedirs(media_dir_path, exist_ok=True) # 确保媒体目录存在

                    # 使用 pypandoc.convert_file 进行转换
                    # to='markdown' 指定输出格式
                    # outputfile 指定输出文件路径
                    # extra_args=['--extract-media', media_dir_path] 传递额外的 pandoc 参数，提取图片到指定目录
                    pypandoc.convert_file(
                        src_file_path,
                        to='markdown',
                        outputfile=target_file_path,
                        extra_args=['--extract-media', media_dir_path]
                    )
                    print(f"pypandoc 转换成功，媒体提取到: {media_dir_path}")

                    # 读取 pypandoc 生成的 Markdown 内容，以便后续处理图片引用
                    with open(target_file_path, "r", encoding="utf-8") as f:
                         markdown_content = f.read()


                elif file_extension == ".pdf":
                     # --- 处理 .pdf 文件使用 MarkItDown ---
                     print(f"正在使用 md_converter 转换: {src_file_path} -> {target_file_path}")
                     # 调用 MarkItDown 转换器进行转换
                     # 假设 MarkItDown.convert 支持 .pdf 文件并能处理其中的图片
                     result = md_converter.convert(src_file_path)
                     # 假设 md_converter.convert 返回的结果有一个 .markdown 属性
                     markdown_content = result.markdown if hasattr(result, 'markdown') else str(result)


                else:
                    # --- 处理其他文件类型使用传入的 md_converter ---
                    # 您可以根据需要扩展这里来处理其他文件类型
                    print(f"正在使用 md_converter 转换未知文件类型: {src_file_path}")
                    # 调用转换器进行转换
                    result = md_converter.convert(src_file_path)
                    markdown_content = result.markdown if hasattr(result, 'markdown') else str(result)


            except FileNotFoundError:
                 # pypandoc 找不到 pandoc 可执行文件时会抛出此异常
                 print(f"错误: 未找到 pandoc 可执行文件。请确保 pandoc 已安装并添加到系统的 PATH 中。")
                 # 这里选择不跳过，因为可能是其他文件类型转换失败，或者希望继续处理已转换的文件
                 # continue # 如果只希望跳过当前文件
                 raise # 选择抛出异常终止整个转换过程，以便修复环境问题
            except RuntimeError as e:
                 # pypandoc 调用 pandoc 失败时可能抛出 RuntimeError
                 print(f"pypandoc 转换文件 {src_file_path} 失败: {str(e)}")
                 continue # 跳过当前文件，处理下一个
            except Exception as e:
                print(f"转换文件 {src_file_path} 失败: {str(e)}")
                continue # 跳过当前文件，处理下一个

            # 如果成功获取了 Markdown 内容，则进行图片引用处理并保存
            if markdown_content is not None:
                 try:
                     # 处理Markdown中的图片引用，特别是对于 .docx 转换后的 Markdown
                     # 将目标文件所在的目录作为 base_dir 传入 process_markdown_images
                     # os.path.dirname(target_file_path) 获取 Markdown 文件所在的目录
                     base_dir_for_images = os.path.dirname(target_file_path)
                     processed_markdown = process_markdown_images(md_converter, markdown_content, base_dir_for_images)

                     # 写入最终的 Markdown 文件
                     with open(target_file_path, "w", encoding="utf-8") as f:
                         f.write(processed_markdown)
                     print(f"Markdown 及其图片描述已保存到: {target_file_path}")

                 except Exception as e:
                     print(f"处理图片引用或写入文件时出错: {str(e)}")
                     # 即使图片处理失败，也尝试保存原始转换的 Markdown 内容？
                     # 这里选择跳过，因为图片描述是多模态理解的关键部分
                     continue # 跳过当前文件


# --- EvidenceProcessPipeline 类开始 ---

class EvidenceProcessPipeline:
    """
    处理原始文档，生成可供 EvidenceLoader 加载的 JSONL 文件。
    流水线包括：格式转换 (含富文本与图片处理) -> 加载 -> 分割 -> 元数据标注 -> 格式化输出。
    """
    def __init__(self):
        # 初始化客户端和工具
        # LLM 客户端用于 MarkItDown，特别是处理图片时的视觉理解
        self.llm_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        # MarkItDown 转换器，配置视觉 LLM 以支持图片内容的理解
        self.md_converter = MarkItDown(llm_client=self.llm_client, llm_model=LLM_MODEL_TRANSFORM)

        # 初始化 Langchain LLM 和 Embedding 模型 (用于标注和可能的 Embedding 生成)
        # 用于元数据标注的 LLM
        self.tagging_llm = ChatOpenAI(
            model=LLM_MODEL_TAGGING,
            base_url=DASHSCOPE_BASE_URL,
            api_key=DASHSCOPE_API_KEY,
            temperature=0.0, # 元数据标注通常需要确定性结果，温度设低
        )
        # Embedding 模型 (如果需要在流水线中生成 embedding)
        # 如果需要，在这里初始化 self.embeddings
        # self.embeddings = DashScopeEmbeddings(
        #     model=EMBEDDING_MODEL,
        #     api_key=DASHSCOPE_API_KEY,
        #     base_url=DASHSCOPE_BASE_URL # DashScope 的 Embedding 模型通常不需要 base_url
        # )


        # 初始化分割器
        # 按 Markdown 标题分割
        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3'),
                ('####', 'Header 4'), ('#####', 'Header 5'), ('######', 'Header 6')
            ]
        )
        # 递归字符分割，用于处理过长或不符合标题结构的片段
        self.recursive_character_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # 您可以根据需要调整片段大小
            chunk_overlap=200, # 您可以根据需要调整重叠大小
            length_function=len,
            # 保留图片描述标签作为分隔符，有助于将图片描述与其相关的文本内容分开或一同处理
            # 注意：seperators 的顺序很重要，会优先按前面的分隔符分割
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
        os.makedirs(RAW_DOCS_DIR, exist_ok=True) # 确保原始文档目录也存在，方便放置文件
        os.makedirs(MARKDOWN_DOCS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(FINAL_EVIDENCE_FILE), exist_ok=True) # 确保最终输出文件所在的目录存在


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
        调用内部实现的 convert_and_save_files 将原始文档转换为 Markdown 格式并保存。
        这个函数内部处理 .docx 和图片的转换与描述生成。
        """
        print(f"开始将原始文档从 '{input_dir}' 转换为 Markdown 到 '{output_dir}'...")
        try:
            # 调用模块内实现的转换函数，实现对 .docx 等富文本和图片的精细处理
            # 这个函数会处理图片提取和 MarkItDown 调用以获取图片描述
            convert_and_save_files(self.md_converter, input_dir, output_dir)
            print("文档转换完成。")
        except FileNotFoundError as e:
            print(f"转换失败: 文件未找到，可能是 pandoc 未安装或不在 PATH 中。错误: {e}")
            # 抛出异常终止整个转换过程，以便用户修复环境问题
            raise e
        except Exception as e:
            print(f"文档转换过程中发生错误: {e}")
            # 抛出异常，表示转换步骤失败
            raise e


    def load_and_split_markdown_docs(self, input_dir: str) -> List[Document]:
        """从 Markdown 文件加载文档并进行分割。"""
        print(f"开始加载并分割 Markdown 文档从 '{input_dir}'...")
        # 使用 TextLoader 并指定 encoding
        # glob="**/*.md" 递归查找所有 .md 文件
        loader = DirectoryLoader(input_dir, glob="**/*.md", show_progress=True, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()
        print(f"加载了 {len(docs)} 个 Markdown 文件。")

        if not docs:
            print("没有加载到任何 Markdown 文档，跳过分割。")
            return []

        doc_splits = []
        print("开始分割文档...")
        for doc in tqdm(docs, desc="分割文档"):
            # 将原始文档按 Markdown 头部分割
            # header_splits 是一个 Document 列表
            header_splits = self.markdown_header_splitter.split_text(doc.page_content)

            # 从原始 Document 复制 metadata 到 header_splits
            for h_split in header_splits:
                 # 保留原始文件的 metadata，如 'source', 'file_path' 等
                 # Langchain 的 split_text 方法通常会复制原始文档的 metadata
                 # 如果没有，可能需要手动复制：h_split.metadata.update(doc.metadata)
                 pass # 假设 split_text 保留了原始 metadata

            # 对 header_splits 进行进一步处理（过滤、递归分割）
            processed_header_splits = []
            for split in header_splits:
                 # 过滤掉过短的片段，这些片段通常意义不大
                 if len(split.page_content.strip()) < 10: # strip() 去除空白字符后再判断长度
                     continue
                 # 对超长片段进行递归分割
                 elif len(split.page_content) > 4000: # 可以根据实际情况调整超长片段的阈值
                     print(f"片段过长 ({len(split.page_content)}), 进行递归分割...")
                     # recursive_character_splitter.split_documents 需要一个 Document 列表作为输入
                     recursive_source_metadata = split.metadata.copy()
                     recursive_splits = self.recursive_character_splitter.split_documents([split])
                     for rec_split in recursive_splits:
                          # Explicitly update metadata for recursive splits
                          rec_split.metadata.update(recursive_source_metadata)

                     processed_header_splits.extend(recursive_splits)
                 else:
                     # 适中长度的片段直接添加
                     processed_header_splits.append(split)

            doc_splits.extend(processed_header_splits)

        print(f"分割后得到 {len(doc_splits)} 个文档片段。")
        return doc_splits

    def tag_document_metadata(self, doc_splits: List[Document]) -> List[Document]:
        """使用 LLM 和预定义 Schema 为文档片段标注元数据。"""
        print("开始标注文档片段元数据...")
        if not doc_splits:
            print("没有文档片段可供标注，跳过标注步骤。")
            return []

        # 使用 tqdm 显示进度
        tagged_splits = []
        try:
            # Langchain 的 transform_documents 函数本身处理对 LLM 的调用和批量处理
            # tqdm 用于显示整体进度
            # 注意：Langchain 的 transform_documents 内部的迭代进度不会被外部的 tqdm 显示
            # 如果需要更细粒度的进度，需要修改 Langchain 内部或自己手动分批处理
            print(f"正在使用 LLM ({self.tagging_llm.model_name}) 进行元数据标注...")
            tagged_splits = self.document_transformer.transform_documents(doc_splits)
            print("元数据标注完成。")

        except Exception as e:
            print(f"元数据标注过程中发生错误: {e}")
            # 标注失败，可能需要检查 LLM 服务或 Schema
            raise e

        # 过滤掉标注失败（metadata 为 None 或 tags 不存在）的片段
        valid_tagged_splits = [
            split for split in tagged_splits
            if split.metadata is not None
        ]
        print(f"成功标注 {len(valid_tagged_splits)} / {len(tagged_splits)} 个文档片段。")

        return valid_tagged_splits


    def format_to_evidence_jsonl(self, tagged_splits: List[Document], output_file: str,
                                 project: Optional[str] = None, collector: Optional[str] = None,
                                 collection_time: Optional[int] = None):
        """
        将带元数据的文档片段格式化为 EvidenceMaterial 并保存为 JSONL 文件。
        """
        print(f"开始格式化并保存证据到 '{output_file}'...")
        if not tagged_splits:
            print("没有带元数据的文档片段，跳过格式化和保存步骤。")
            return

        evidences: List[EvidenceMaterial] = []
        for split in tagged_splits:
            try:
                file_path = split.metadata.get("source") # Get 'source' from metadata
                source_filename = os.path.basename(file_path) if file_path else "unknown"

                tags = split.metadata.copy() # 创建 metadata 的副本，避免修改原始对象
                keys_to_remove_from_tags = ["source", "heading", "file_path"] # 移除 source, heading, file_path 等内部或非tag字段
                for key in keys_to_remove_from_tags:
                    tags.pop(key, None) # 使用 pop 的第二个参数 None，即使 key 不存在也不会报错

                # Embedding 生成 (如果需要)
                # embedding_vector = None
                # if hasattr(self, 'embeddings') and self.embeddings:
                #      try:
                #           embedding_vector = self.embeddings.embed_query(split.page_content)
                #      except Exception as e:
                #           print(f"警告: 生成 embedding 失败，跳过。片段内容开头：{split.page_content[:50]}... 错误: {e}")
                formatted_collection_date = None
                if collection_time is not None:
                    try:
                        dt_object = datetime.fromtimestamp(collection_time)
                        formatted_collection_date = dt_object.strftime('%Y-%m-%d')
                    except Exception as e:
                        print(f"Warning: Failed to format collection time {collection_time}. Error: {e}")
                        formatted_collection_date = str(collection_time) # Fallback to string representation

                evidence_data = {
                    "content": split.page_content,
                    "tags": tags,
                    "source": source_filename,
                    "project": project,
                    "collection_time": formatted_collection_date,
                    "collector": collector,
                    # "embedding": embedding_vector
                }
                evidences.append(EvidenceMaterial(**evidence_data))
            except Exception as e:
                source_info = split.metadata.get("source", "未知文件")
                print(f"警告：格式化文档片段时出错，跳过。来自文件: {source_info}，片段内容开头：{split.page_content[:100]}... 错误: {e}")
                continue


        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 使用 tqdm 显示写入进度
                for evidence in tqdm(evidences, desc="写入 JSONL"):
                    # 调用 model_dump_json()，不再传递 ensure_ascii=False
                    # 如果需要 ensure_ascii=False 的效果，可以在写入时手动 json.dumps
                    # 但 model_dump_json 通常会处理 unicode 字符，直接调用即可
                    # Pydantic v2+ 的 model_dump_json() 返回的是字符串
                    f.write(evidence.model_dump_json() + '\n')

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

        # 步骤 1: 文档格式转换 (调用内部实现的 convert_and_save_files)
        # 此步骤处理 .docx 等富文本和图片的多模态转换
        self.transform_documents_to_markdown(RAW_DOCS_DIR, MARKDOWN_DOCS_DIR)


        # 步骤 2&3: 加载转换后的 Markdown 文档并进行分割
        markdown_splits = self.load_and_split_markdown_docs(MARKDOWN_DOCS_DIR)

        if not markdown_splits:
            print("没有找到可处理的 Markdown 文档片段，流水线结束。")
            return

        # 4. 元数据提取和标注
        # 使用 LLM 和 Schema 对文档片段进行语义标注
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
            # 如果生成了 embedding，这里也需要传递 embedding_vector
        )

        print("证据处理流水线执行完毕。")


# 示例用法
if __name__ == "__main__":
    # 确保您的 .env 文件中设置了 DASHSCOPE_BASE_URL 和 DASHSCOPE_API_KEY
    # 确保安装了 pypandoc 和 pandoc
    # 确保创建了 Docs/ 目录并将原始文档放入其中
    # 确保创建了 DocumentPipeline/evidence_schema.json 文件

    pipeline = EvidenceProcessPipeline()

    # 在运行时指定参数
    # 您可以从命令行参数、配置文件或数据库中获取这些信息
    my_project = "网络安全评估项目"
    my_collector = "张三" # 或者从某个配置或命令行参数获取
    my_collection_time = int(time.time()) # 如果需要指定一个固定的时间，比如任务开始时间
    # 如果不指定 collection_time，run 方法将使用当前时间

    print(f"准备运行流水线，指定项目: '{my_project}', 采集人: '{my_collector}'")

    # 如果需要指定采集时间，可以这样调用：
    pipeline.run(project=my_project, collector=my_collector, collection_time=my_collection_time)