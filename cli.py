
import traceback
from Core import NaviSearchCore

class NaviSearchCLI:
    def __init__(self, search_core: NaviSearchCore):
        self.core = search_core

    # 本次新增加的选项，用于展示数据库的信息


    def _display_welcome_message(self):
        print("\n=== 交互式知识库查询 ===")
        print("输入命令或直接输入查询内容。输入 /help 查看更多命令。")
        print("-" * 20)

    def _display_help(self):
        print("\n=== 操作指引 ===")
        print("可用命令:")
        print("  /query <内容>  - 执行自然语言查询 (直接输入内容也可)")
        print("  /tags          - 查看当前已激活的过滤标签")
        print("  /tag <标签1>,<标签2>,... - 添加一个或多个过滤标签 (逗号分隔)")
        print("  /tags clear    - 清空所有已激活的过滤标签")
        print("  /collection    - 查看当前使用的集合")
        print("  /collection count- 查看当前集合中的实体数量")
        print("  /collection schema - 查看当前集合的模式")
        print("  /collection list - 查看所有集合")
        print("  /collection use <collection name> - 使用指定的集合")
        print("  /help          - 显示此帮助信息")
        print("  /bye           - 退出程序")
        print("\n示例:")
        print("  > 如何测试电池寿命")
        print("  > /tag 电池,寿命")
        print("  > /tags")
        print("  > /tags clear")
        print("=" * 20 + "\n")

    def _handle_collection_command(self, user_input):
        command_part = user_input.split(" ", 1)
        command = command_part[0]
        content = command_part[1].strip() if len(command_part) > 1 else ""
        collection_name = command_part[2].strip() if len(command_part) > 2 else ""

        if command == '/collection' and not content:
            current_collection = self.core.get_current_collection()
            print(f"当前使用的集合: {current_collection}")

        elif command == '/collection' and content == 'count':
            collection_name = self.core.get_current_collection()
            count = self.core.collection.num_entities
            print(f"集合 {collection_name} 中的实体数量: {count}")

        elif command == '/collection' and content == 'schema':
            schema = self.core.schema
            for field in schema.fields:
                print(f"Field name: {field.name}")
                print(f"  dtype: {field.dtype}")
                print(f"  description: {field.description}")
                print(f"  is_primary: {field.is_primary}")
                print(f"  auto_id: {field.auto_id}")
                print("----------")

        elif command == '/collection' and content == 'list':
            list_collection_response = self.core.list_collections()
            print("所有集合:")
            for collection in list_collection_response.get('collections', ''):
                print(f"- {collection}")

        elif command == '/collection' and content == 'use' and collection_name:
            msg = self.core.use_collection(collection_name)
            print(msg)
    def _handle_tag_command(self, user_input):
        command_part = user_input.split(" ", 1)
        command = command_part[0]
        content = command_part[1].strip() if len(command_part) > 1 else ""

        if command == '/tags' and not content:
            active_tags = self.core.get_active_tags()
            if active_tags:
                print(f"当前激活标签: {active_tags}")
            else:
                print("当前没有激活的标签。")

        elif command == '/tags' and content == 'clear':
            msg = self.core.clear_tags()
            print(msg)

        elif command == '/tag' or (command == '/tags' and content and content != 'clear'):
            tag_content_to_parse = ""
            if command == '/tag' and len(command_part) > 1:
                tag_content_to_parse = command_part[1].strip()
            elif command == '/tags' and content and content != 'clear':
                tag_content_to_parse = content

            msg = self.core.add_tags(tag_content_to_parse)
            print(msg)
            print(f"当前激活标签: {self.core.get_active_tags()}")

        else:
            print("无效的标签命令。使用 /help 查看用法。")

    def _display_formatted_results(self, formatted_result):
        results_list = formatted_result['results']
        rec_tags = formatted_result['rec_tags']

        if not results_list:
            print("没有找到匹配的结果。\n")
            return

        print(f"\n找到 {len(results_list)} 个相关结果:")
        for i, res in enumerate(results_list):
            print(f"\n--- 结果 {i+1} ---")
            print(f"内容: {res['content']}")
            print(f"标签: {res['tags']}")
        print("-" * 20 + "\n")

        if rec_tags:
            print(f"推荐过滤标签: {rec_tags}")
            print("-" * 20 + "\n")

    def run(self):
        self._display_welcome_message()
        while True:
            try:
                user_input = input("> ").strip()
                if not user_input:
                    continue

                if user_input == '/bye':
                    print("收到退出命令。")
                    break
                elif user_input == '/help':
                    self._display_help()
                elif user_input.startswith(('/collection', '/collections')):
                    self._handle_collection_command(user_input)
                elif user_input.startswith(('/tag', '/tags')):
                    self._handle_tag_command(user_input)
                else :
                    if user_input.startswith('/query '):
                        query_text = user_input[len('/query '):].strip()
                    else:
                        query_text = user_input.strip()
                    print(f"正在搜索: '{query_text}' (过滤标签: {self.core.get_active_tags() or '无'})")
                    query_result = self.core.perform_search(query_text)
                    if query_result['status'] == 'success':

                        ranked_records = query_result['ranked_records']
                        ranked_tags = query_result['ranked_tags']

                        for index, record in enumerate(ranked_records):
                            print(f"--- 结果 {index+1} ---")
                            print(record.get("content"[:300]))
                            print(record.get("tags"))


                        print(f"推荐过滤标签: {ranked_tags[:20]}")
                        print("*"*50)
                    else:
                        print(f"\n搜索过程中发生错误: {query_result['message']}\n")

            except EOFError:
                print("\n检测到输入结束 (EOF)，退出。")
                break
            except KeyboardInterrupt:
                print("\n收到中断信号，退出。")
                break
            except Exception as e:
                print(f"\n处理输入时发生意外错误: {e}")
                traceback.print_exc()
                break

if __name__ == "__main__":
    search_core = NaviSearchCore(init_collection=False,tags_design_path = "Data/Tags/tags_design.json")  # 实例化 NaviSearchCore 类
    cli = NaviSearchCLI(search_core)  # 实例化 NaviSearchCLI 类
    cli.run()  # 运行 CLI 应用