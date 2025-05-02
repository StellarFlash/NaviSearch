from markitdown import MarkItDown
from openai import OpenAI

client = OpenAI(
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = "sk-1fbb0eb503664e728ebeabe3faa97d62"
)
md = MarkItDown(llm_client=client, llm_model="qwen-vl-max")
result = md.convert("Data/Corpus/广西科技师范学院无线（WIFI）网络系统项目建设施工方案1228.pdf")
print(result.text_content)