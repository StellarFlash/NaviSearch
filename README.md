# NaviSearch: 企业级知识库智能搜索引擎

NaviSearch 是一个专注于存储和检索经过良好标注的企业内部文档的智能搜索引擎。它旨在通过增强的重排序机制，特别是利用文档的元数据标签（tags），来提供比传统向量搜索更准确、更相关的搜索结果。

## 项目定位与价值

在企业知识管理领域，快速准确地找到所需信息至关重要。NaviSearch 的定位介于基础的向量搜索引擎和复杂的深度研究系统之间：

* **克服基础向量搜索的局限性**: 传统向量搜索可能因缺乏领域专业知识的嵌入模型而导致搜索结果相关性不高。NaviSearch 通过利用精细标注的文档 `tags` 来优化排序，从而提升领域适应性。
* **简化复杂性**: 相较于需要大量基础设施改造、开发周期长且可能存在不可控因素的深度研究型智能体系统，NaviSearch 提供了一种更轻量级、更易于实施和维护的解决方案。
* **核心优势**:
    * **基于标签的智能重排序**: NaviSearch 的核心在于其重排序阶段，它充分利用文档 `tags` 信息来提升目标文档的排名。
    * **Query-Free 排序信号**: 它使用 `tags` 作为查询无关 (query-free) 的排序信号，这意味着这些信号可以预计算、缓存和复用，从而在效率上优于需要实时计算的查询相关 (query-dependent) 方法。

## 主要特性

* **管理端 (Admin)**:
    * **知识库管理**: 提供 Milvus Collection 的创建、删除、初始化、索引构建和加载等功能。 ([AdminCore.py](AdminCore.py), [AdminFastAPI.py](AdminFastAPI.py))
    * **文档入库**: 支持单条和批量插入带有 `content`、`tags` 和 `embedding` 的文档。若未提供 `embedding`，系统可尝试自动生成。 ([AdminCore.py](AdminCore.py), [AdminFastAPI.py](AdminFastAPI.py))
    * **WebUI 管理界面**: 提供基于 Gradio 的可视化管理界面，方便管理员操作和调试。 ([AdminWebui.py](AdminWebui.py))
* **用户访问与搜索端 (Visitor)**:
    * **核心搜索逻辑**: 封装了从初步检索到基于标签重排序的完整搜索流程。 ([VisitorCore.py](VisitorCore.py), [SearchEngine.py](SearchEngine.py))
    * **两种搜索模式**:
        * **Standard Mode**: 传统的检索后基于用户激活标签进行重排序。
        * **Agentic Mode**: 实验性的智能搜索模式，利用 LLM 迭代生成和优化过滤标签，以期获得更精确的结果。
    * **动态标签过滤**: 用户可以动态添加或清除激活的 `tags`，以辅助和优化搜索过程。
    * **API 服务**: 提供 FastAPI 接口，方便集成到其他应用或自动化流程中。 ([VisitorFastAPI.PY](VisitorFastAPI.PY))
    * **WebUI 调试界面**: 提供基于 Gradio 的搜索界面，用于测试和验证搜索效果，尤其适用于 RAG (Retrieval Augmented Generation) 系统的召回验证。 ([VisitorWebui.py](VisitorWebui.py))
* **工具集 (`utils.py`)**:
    * **Embedding 生成**: 对接外部服务（如百炼）生成文本的向量表示。
    * **LLM 交互**: 对接外部 LLM 服务，用于 `agentic` 模式下的标签选择和可能的辅助打标功能。
    * **辅助打标**: 提供了基于 LLM 为文档内容生成语义标签的功能。

## 工作原理简述

1.  **文档入库与预处理**:
    * 管理员通过 Admin 接口或 WebUI 将文档（包含内容和预先标注的 `tags`）存入 Milvus。
    * 系统为文档内容生成向量 `embedding`。
    * `tags` 作为重要的元数据被一同存储，并成为后续重排序的关键。

2.  **搜索执行**:
    * 用户通过 Visitor API 或 WebUI 发起查询。
    * **初步检索**: `SearchEngine` 使用查询文本的 `embedding` 在 Milvus 中进行向量相似度搜索，召回一批候选文档。
    * **重排序**:
        * 在 **Standard Mode** 下，`SearchEngine` 根据用户指定的或会话中激活的 `filter_tags`，对候选文档进行重排序。排序策略可以是 "filtering"（文档必须包含所有指定标签）或 "ranking"（文档包含至少一个指定标签），并根据匹配标签数量排序。
        * 在 **Agentic Mode** 下，`VisitorCore` 会与 LLM (`get_filter` in `utils.py`) 交互，根据当前查询、已选标签和候选文档的特征，迭代地推荐新的过滤标签组合，然后使用这些标签进行多轮次的过滤和重排序，直至结果收敛。
    * **结果返回**: 返回排序后的文档列表以及推荐的相关标签。

## 生态位与未来

NaviSearch 的设计目标是为企业提供一个高效、准确且易于集成的知识搜索引擎。在完成召回测试和功能验证后，它计划被嵌入到更广泛的企业自动化系统中，以提升知识利用效率和团队协作能力。

其对高质量元数据标注的依赖，也提示了在组织内部推行知识工程和标准化文档标注的重要性。

## 如何开始 (Placeholder)

*(这部分可以根据实际部署情况填写)*

1.  **环境配置**:
    * Python 3.x
    * Milvus (及相应的 Python SDK `pymilvus`)
    * OpenAI (或其他兼容的 Embedding 和 LLM 服务)
    * `requirements.txt` (需要创建)
2.  **配置环境变量**:
    * `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`
    * `LLM_MODEL` (用于 `get_response` 和 `get_filter`)
    * Milvus 连接参数 (可在 `AdminCore.py` 和 `VisitorCore.py` 中配置默认值或修改为从环境变量读取)
3.  **启动服务**:
    * 启动 Milvus 服务。
    * 启动 Admin FastAPI 服务: `python AdminFastAPI.py`
    * 启动 Visitor FastAPI 服务: `python VisitorFastAPI.PY`
    * 启动 Admin WebUI (可选): `python AdminWebui.py`
    * 启动 Visitor WebUI (可选): `python VisitorWebui.py`

## 模块说明

* `AdminCore.py`: 管理端核心逻辑，封装 Milvus Collection 和文档操作。
* `AdminFastAPI.py`: 管理端 API 服务。
* `AdminWebui.py`: 管理端 Gradio Web 界面。
* `VisitorCore.py`: 用户访问端核心逻辑，封装搜索和标签管理。
* `VisitorFastAPI.PY`: 用户访问端 API 服务。
* `VisitorWebui.py`: 用户访问端 Gradio Web 界面。
* `SearchEngine.py`: 封装 Milvus 检索和重排序算法。
* `utils.py`: 工具函数，包括 Embedding 生成、LLM 调用等。
* `Tagger/SemanticTagger.py`: (在 `VisitorCore.py` 中被导入但未在核心流程中明确使用，推测与辅助打标相关)