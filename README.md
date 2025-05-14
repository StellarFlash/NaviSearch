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

## 演示案例

我们准备了一个简单的演示案例，展示了如何使用NaviSearch完成实际的任务。

### 文档预处理

使用DocumentPipeline对文档进行预处理，包括多模态理解、格式转化、文档切分与元数据标注。

可以修改evidence_schema.json来适应不同领域的任务。

```bash
python DocumentPipeline/evidence_pipeline.py
```

### 运行评估系统

使用AssementSystem来生成评估报告。AssementSystem使用AssementSystem/assement_schema.json来定义评估任务，从NaviSearch中召回文档进行评估。

```bash
python AssementSystem/assement_engine.py
```

评估结论保存在AssementSystem/assement_report.json中。

```python
{
    "assessment_results": [
        {
            "spec_id": "3.1.3.1.b",
            "spec_content": "3.1.3.1 5G基站或核心网支持对5G终端进行访问控制，仅允许特定的行业终端进入5G专网，限定条件包括但不限于终端标识（IMSI/SUPI）、终端位置（CGI、TAI）、IMEI黑名单、流量限制、机卡绑定等。",
            "evidence_search_params": {
                "query_text": "5G核心网是否支持通过二次认证系统配置流量限制规则（如协议、带宽、时间段等），并验证在异常流量行为下规则的有效性。",
                "filter_tags": [
                    "LST IPSECPOLICY",
                    "分流策略",
                    "访问控制(Access Control)",
                    "securityIndication.integrityProtectionIndication=required",
                    "LST DDOS",
                    "GTP-U",
                    "AMF"
                ],
                "terminated": false
            },
            "conclusion": {
                "judgement": "符合",
                "comment": "根据材料1和材料6，5G专网支持对终端进行访问控制，包括通过IMSI/SUPI、IMEI黑名单等方式限制终端接入，并可通过配置规则限制终端的流量行为。此外，材料6提到启用了CPE白名单功能，仅允许特定设备接入，进一步验证了对终端访问控制的能力。",
                "evidence": [
                    {
                        "source": "458013357677764955",
                        "content": "1.  5G基站或核心网支持对5G终端进行访问控制，仅允许特定的行业终端进入5G专网，限定条件包括但不限于终端标识（IMSI/SUPI）、终端位置（CGI、TAI）、IMEI黑名单、流量限制、机卡绑定等。  \n> 确认核心网侧是否针对该项目终端的接入做了额外配置；说明该项目网络中是否部署二次认证等安全能力，并提供证明材料。  \n2.  5G\n专网应开启NAS层机密和完整性保护，能够对NAS信令进行机密性和完整性保护。（如果企业不具备或未开启加密机制，需要提供说明材料。）  \n> 核心网侧确认是否在AMF上开启NAS加密  \n3.  5G专网N4接口是否支持物理隔离，或者具备机密性保护、完整性保护和抗重放保护机制。  \n> 提供UPF侧LST IPSECPROPOSAL、LST IKECFG的查询截图  \n4.  N4接口应支持双向认证能力或仅允许指定IP的SMF和UPF互访。如使用NRF进行注册和授权，则未在NRF中合法注册和授权的SMF无法访问UPF。  \n> 确认是否通过NRF进行SMF和UPF的对接；若采取IP配置互通，则于UPF侧提供LST\n> CPNODEID和LST CPACCESSLISTFUNC、LST CPACCESSLIST的截图。\n>\n> 说明是否存在双向认证机制。  \n5.  边缘UPF应支持启停/链路中断告警，当UPF重启/断电/链路中断后，维护终端/网管上均能生成告警记录。  \n> 提供UPF告警日志记录截图  \n6.  UPF应支持信令面/用户面流量控制机制，以确保其在收到大量攻击报文时不会产生异常。  \n> 提供UPF LST LBFC， LST DDOS的查询结果截图\n>\n> ![D:\\\\资料文件整理\\\\测评\\\\山西移动\\\\临汾台头\\\\5G截图\\\\lbfc.PNG](Data\\Markdown\\5G行业应用安全风险评估资料需求v2-来宾_media/media/image40.png){width=\"5.768055555555556in\"\n<image_discription>\n> # Description:\n> 通用维护(AIt+C) 操作记录(AIt+R) 帮助信息(AIt+N)\n> 操作结果如下：\n> 流控开关 = 打开\n> (结果个数 = 1)\n> ---- END\n> 区自动滚动 清除报告(F6) 启动重定向\n</image_discription>  \n> height=\"1.1548611111111111in\"}\n>\n> ![D:\\\\资料文件整理\\\\测评\\\\山西移动\\\\临汾台头\\\\5G截图\\\\ddos.PNG](Data\\Markdown\\5G行业应用安全风险评估资料需求v2-来宾_media/media/image41.png){width=\"5.768055555555556in\"\n<image_discription>\n> # Description:\n> 通用维护(AIt+C) 操作记录(Alt+R) 帮助信息(Alt+N)\n> 结果如下：\n> DDos防护阀值 = 100\n> (结果个数 = 1)\n> ----\n> END\n> 区自动滚动 清除报告(F6) 启动重定向\n</image_discription>  \n> height=\"1.2013888888888888in\"}  \n7.  边缘UPF应支持IPSec加密、访问控制等安全功能。  \n> 提供UPF 的LST ACL、LST SECPOLICY、DSP ACLRULEADV4的查询结果截图  \n8.  UPF应支持分流策略安全机制，可进行分流策略的配置、冲突检测和告警。  \n> 提供UPF的LST COLOCATEDLBO， DSP COLLISIONCHECK，LST\n> QOSAPPLICATION的查询结果截图。对于有多个参数可选的，提供所有可选参数的查询结果。  \n9.  UPF应支持对UE的互访限制和访问控制，支持UE之间互访策略设置、DNN\nACL策略配置，支持禁止园区私网主动访问运营商网络。  \n> 提供UPF的LST APNUEMUTACC和LST GLOBUEMUTACC、，LST ACLRULEADV4，LST\n> ACLBINDAPN的查询结果截图。  \n10. 应从计算、存储和网络资源等方面加强网络虚拟化基础设施安全保障，对网络虚拟设施的所有操作应纳入统一管理平台，实现集中访问控制和安全审计。  \n> 提供登录网元时4A接入界面截图，并说明4A的访问控制和安全审计功能。  \n11. 应根据不同虚拟机功能合理划分内部安全域，做好域间隔离和访问控制。  \n> 提供UPF底层Fusionsphere的VPC/VDC等逻辑隔离和访问控制功能截图  \n12. 应定期对物理/虚机操作系统、虚拟化软件、第三方开源软件实施安全加固。  \n> 提供系统物理/虚机操作系统、虚拟化软件、第三方开源软件定期安装补丁或更安全的版本等的截图，或相关台账记录等。  \n13. 应支持切片网元隔离，确保非共享网元只出现在一个切片中。  \n> 提供UPF等下沉网元的LST SNSSAI、LST SLICEINSTINFO查询截图。  \n14. 查看虚拟机是否专用、物理机是否专用，以保障切片资源隔离。  \n> 提供UPF等下沉网元底层Fusionsphere的资源分配情况，说明虚机提供的功能以及物理机位置。"
                    },
                    {
                        "source": "458013357677765022",
                        "content": "针对有高可靠性要求的区域，可以考虑CPE\n1+1冗余覆盖的方案，具体实施原则和注意事项包括：  \n冗余备份的两台CPE的SSID和密码设置相同，WiFi信道设置为不同；  \n非冗余CPE之间采用不同的密码，确保非区域终端不能随便接入；  \n启用CPE白名单功能，仅允许名单内设备接入上网；  \nCPE将支持5G和有线备份功能；"
                    }
                ]
            },
            "status": "success",
            "error_message": null
        },
        {
            "spec_id": "3.1.1.1.a",
            "spec_content": "3.1.1.1 具备5G功能的终端和为行业终端提供接入服务的5G接入网关设备，应支持3GPP标准的认证机制，建议终端选择配备二次认证、安全芯片、GBA认证或AKMA认证机制。",
            "evidence_search_params": {
                "query_text": "请提供具备5G功能的终端及接入网关设备支持3GPP标准认证机制（包括二次认证、安全芯片、GBA认证和AKMA认证）的技术文档，重点说明其在行业应用中的实现方式、配置细节与合规性验证方法。",
                "filter_tags": [
                    "认证机制",
                    "5G终端",
                    "接入网关",
                    "GBA认证",
                    "AKMA认证",
                    "安全芯片",
                    "二次认证",
                    "3GPP标准",
                    "访问控制(Access Control)",
                    "机密性和完整性保护算法"
                ],
                "terminated": false
            },
            "conclusion": {
                "judgement": "符合",
                "comment": "根据材料2和材料8，5G终端和接入网关支持3GPP标准的认证机制，并提到了二次认证等安全能力。材料2明确要求提供5G终端和接入网关支持3GPP认证机制及加密算法的说明文档；材料8中提到核心网侧部署了针对终端的访问控制和二次认证机制。",
                "evidence": [
                    {
                        "source": "458013357677764937",
                        "content": "1.  提供5G行业应用设计文档或其他说明材料，内容应至少包含5G行业应用名称，所属行业、业务场景、预计应用规模（开卡数量、5G终端数量、总终端数量等）、行业应用类型、设备部署位置及规划、第三方云服务使用情况、切片使用情况及类型、专网模式、行业应用简介、网络拓扑图、业务流量和特性、各项资产和运维责任归属情况等。[（计划部、政企、客响分别提供设计方案文档）]{.mark}  \n2.  说明5G终端和接入网关应支持3GPP标准的认证机制，并支持3GPP标准中要求的机密性和完整性保护算法（如128-NEA1,128-NEA2,128-NEA3,128-NIA1,128-NIA2和128-NIA3）  \n> 提供5G终端和接入网关等的认证机制和加密算法的说明文档。[（政企部）]{.mark}  \n3.  说明该项目是否存在多方合作。若存在合作则应提供合作方式合规性评估记录文档和合作企业的安全保障能力评估记录文档。[（政企部，可以参考以下附件中任何一个来做）]{.mark}  \n![](Data\\Markdown\\5G行业应用安全风险评估资料需求v2-来宾_media/media/image28.emf)![](Data\\Markdown\\5G行业应用安全风险评估资料需求v2-来宾_media/media/image29.emf)  \n4.  说明该项目是否采用了第三方云服务。若存在第三方云服务，则应提供针对第三方云服务商的身份与资质证明材料；针对第三方云计算应提供安全保障技术和安全管理制度文档。[（政企部）]{.mark}  \n5.  说明新建5G应用平台物理安全的安全保障措施，提供安全技术措施文档和安全管理制度文档，明确应用服务器、机房、节点的地理位置，应具备一定的管理措施和技术措施确保其物理环境安全。  \n> [（信安部提供制度）]{.mark}\n>\n> ![](Data\\Markdown\\5G行业应用安全风险评估资料需求v2-来宾_media/media/image30.emf)  \n![](Data\\Markdown\\5G行业应用安全风险评估资料需求v2-来宾_media/media/image31.png){width=\"5.764583333333333in\"\nheight=\"3.9791666666666665in\"}"
                    },
                    {
                        "source": "458013357677764955",
                        "content": "1.  5G基站或核心网支持对5G终端进行访问控制，仅允许特定的行业终端进入5G专网，限定条件包括但不限于终端标识（IMSI/SUPI）、终端位置（CGI、TAI）、IMEI黑名单、流量限制、机卡绑定等。  \n> 确认核心网侧是否针对该项目终端的接入做了额外配置；说明该项目网络中是否部署二次认证等安全能力，并提供证明材料。  \n2.  5G\n专网应开启NAS层机密和完整性保护，能够对NAS信令进行机密性和完整性保护。（如果企业不具备或未开启加密机制，需要提供说明材料。）  \n> 核心网侧确认是否在AMF上开启NAS加密  \n3.  5G专网N4接口是否支持物理隔离，或者具备机密性保护、完整性保护和抗重放保护机制。  \n> 提供UPF侧LST IPSECPROPOSAL、LST IKECFG的查询截图  \n4.  N4接口应支持双向认证能力或仅允许指定IP的SMF和UPF互访。如使用NRF进行注册和授权，则未在NRF中合法注册和授权的SMF无法访问UPF。  \n> 确认是否通过NRF进行SMF和UPF的对接；若采取IP配置互通，则于UPF侧提供LST\n> CPNODEID和LST CPACCESSLISTFUNC、LST CPACCESSLIST的截图。\n>\n> 说明是否存在双向认证机制。  \n5.  边缘UPF应支持启停/链路中断告警，当UPF重启/断电/链路中断后，维护终端/网管上均能生成告警记录。  \n> 提供UPF告警日志记录截图  \n6.  UPF应支持信令面/用户面流量控制机制，以确保其在收到大量攻击报文时不会产生异常。  \n> 提供UPF LST LBFC， LST DDOS的查询结果截图\n>\n> ![D:\\\\资料文件整理\\\\测评\\\\山西移动\\\\临汾台头\\\\5G截图\\\\lbfc.PNG](Data\\Markdown\\5G行业应用安全风险评估资料需求v2-来宾_media/media/image40.png){width=\"5.768055555555556in\"\n<image_discription>\n> # Description:\n> 通用维护(AIt+C) 操作记录(AIt+R) 帮助信息(AIt+N)\n> 操作结果如下：\n> 流控开关 = 打开\n> (结果个数 = 1)\n> ---- END\n> 区自动滚动 清除报告(F6) 启动重定向\n</image_discription>  \n> height=\"1.1548611111111111in\"}\n>\n> ![D:\\\\资料文件整理\\\\测评\\\\山西移动\\\\临汾台头\\\\5G截图\\\\ddos.PNG](Data\\Markdown\\5G行业应用安全风险评估资料需求v2-来宾_media/media/image41.png){width=\"5.768055555555556in\"\n<image_discription>\n> # Description:\n> 通用维护(AIt+C) 操作记录(Alt+R) 帮助信息(Alt+N)\n> 结果如下：\n> DDos防护阀值 = 100\n> (结果个数 = 1)\n> ----\n> END\n> 区自动滚动 清除报告(F6) 启动重定向\n</image_discription>  \n> height=\"1.2013888888888888in\"}  \n7.  边缘UPF应支持IPSec加密、访问控制等安全功能。  \n> 提供UPF 的LST ACL、LST SECPOLICY、DSP ACLRULEADV4的查询结果截图  \n8.  UPF应支持分流策略安全机制，可进行分流策略的配置、冲突检测和告警。  \n> 提供UPF的LST COLOCATEDLBO， DSP COLLISIONCHECK，LST\n> QOSAPPLICATION的查询结果截图。对于有多个参数可选的，提供所有可选参数的查询结果。  \n9.  UPF应支持对UE的互访限制和访问控制，支持UE之间互访策略设置、DNN\nACL策略配置，支持禁止园区私网主动访问运营商网络。  \n> 提供UPF的LST APNUEMUTACC和LST GLOBUEMUTACC、，LST ACLRULEADV4，LST\n> ACLBINDAPN的查询结果截图。  \n10. 应从计算、存储和网络资源等方面加强网络虚拟化基础设施安全保障，对网络虚拟设施的所有操作应纳入统一管理平台，实现集中访问控制和安全审计。  \n> 提供登录网元时4A接入界面截图，并说明4A的访问控制和安全审计功能。  \n11. 应根据不同虚拟机功能合理划分内部安全域，做好域间隔离和访问控制。  \n> 提供UPF底层Fusionsphere的VPC/VDC等逻辑隔离和访问控制功能截图  \n12. 应定期对物理/虚机操作系统、虚拟化软件、第三方开源软件实施安全加固。  \n> 提供系统物理/虚机操作系统、虚拟化软件、第三方开源软件定期安装补丁或更安全的版本等的截图，或相关台账记录等。  \n13. 应支持切片网元隔离，确保非共享网元只出现在一个切片中。  \n> 提供UPF等下沉网元的LST SNSSAI、LST SLICEINSTINFO查询截图。  \n14. 查看虚拟机是否专用、物理机是否专用，以保障切片资源隔离。  \n> 提供UPF等下沉网元底层Fusionsphere的资源分配情况，说明虚机提供的功能以及物理机位置。"
                    }
                ]
            },
            "status": "success",
            "error_message": null
        }
    ],
    "statics": {
        "compliant": 2,
        "non_compliant": 0,
        "not_applicable": 0,
        "not_processed": 0
    }
}
```