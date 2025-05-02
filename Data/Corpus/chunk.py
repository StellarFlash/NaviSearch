import re
import json

def extract_assessment_specs(input_file, output_file):
    # 正则表达式匹配评估规范起始行（如：3.1.1.1 ...）
    pattern = re.compile(r'^(\d+\.\d+\.\d+\.\d+)(.*)')

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    current_section = None

    for line in lines:
        # 判断是否为新的评估规范起始行
        if pattern.match(line.lstrip(' ')):  # 允许行首存在缩进
            if current_section is not None:
                results.append(current_section)
            current_section = line.rstrip('\n')  # 初始化当前规范内容
        else:
            if current_section is not None:
                current_section += '\n' + line.rstrip('\n')  # 追加当前行内容

    # 添加最后一个规范
    if current_section is not None:
        results.append(current_section)

    # 将结果写入 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for content in results:
            obj = {
                "content": content,
                "tags": [],
                "embedding": []
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

# 调用函数
extract_assessment_specs('5G行业应用安全评估测试指引.md', '5G行业应用安全评估测试指引.jsonl')