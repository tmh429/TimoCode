# TimoCode 项目开发计划

## 1. 项目定位与目标

### 1.1 项目介绍

参考开源项目 [Pi Coding Agent](https://github.com/badlogic/pi-mono)，使用 Python 从零构建一个轻量级终端 Coding Agent。

项目应具备自主理解编程任务、读取和修改项目代码、执行命令、分析错误、迭代修复的能力，并支持持久化会话、上下文压缩和任务效果评测。

### 1.2 目标

**工程目标：**
- 理解并实现 Agent Loop 与 Tool Calling。
- 构建支持代码读取、搜索、编辑、命令执行的工具系统。
- 实现多轮交互、会话持久化、上下文管理。
- 提供命令执行安全控制、异常恢复和可观测性。
- 实现可实际使用的终端 Coding Agent。

**算法扩展目标：**
- 构建 Code Agent 任务评测框架。
- 保存工具调用轨迹与模型决策数据。
- 后续接入 MiniMind，开展 Tool Calling SFT 与效果评测。

### 1.3 开发原则

1. 使用 Python 作为主要开发语言，不要求先学习 TypeScript。
2. 参考 Pi 的架构与设计思想，而非机械翻译源码。
3. 优先使用原生 Python、asyncio、Pydantic 等技术实现核心能力。
4. 第一版不依赖 LangChain、LangGraph 等高级 Agent 框架。
5. 允许使用 Codex 辅助编程，但每个核心模块都需要理解其原理。
6. 每个阶段完成后必须测试验收，再进入下一阶段。
7. 优先完成 MVP，不提前开发复杂 UI、多 Agent 或大型插件系统。

---

## 2. 技术栈

| 分类 | 推荐技术 |
|---|---|
| 开发语言 | Python 3.11+ |
| LLM 接口 | OpenAI-compatible API |
| 并发与异步 | asyncio |
| 数据模型 | Pydantic |
| CLI | Typer + Rich |
| 会话持久化 | JSONL |
| 可选数据存储 | SQLite |
| Token 统计 | 模型 tokenizer / tiktoken |
| 自动化测试 | pytest |
| 环境隔离 | Docker |
| 模型训练扩展 | PyTorch、MiniMind |

先实现简单、清晰的模块接口，避免过早引入复杂技术。

---

## 3. 分阶段开发路线

### Phase 0：环境准备与 Pi 架构学习

**学习内容**

- LLM API 的消息结构与请求流程。
- Function Calling / Tool Calling 原理。
- Agent Loop 的工作机制。
- Pi 的 Agent Runtime、Tools、Session、Compaction 设计。
- Python asyncio、异步迭代器及流式响应。

**开发任务**

1. 初始化 Python 项目与 Git 仓库。
2. 配置依赖管理、环境变量、测试环境。
3. 阅读 Pi 官方架构文档和核心源码。
4. 绘制自己的 Code Agent 架构图。
5. 确定 Agent、LLM Provider、Tool、Session 等模块的接口。

**参考源码**
- `packages/ai`
- `packages/agent`
- `packages/coding-agent`

**验收标准**

能够解释 Pi 的核心运行流程，并确定自己的 Agent 架构与模块边界。

### Phase 1：LLM 接入与 Agent Loop（核心）

**学习内容**

- Chat Completion 与流式响应。
- Tool Schema、工具调用参数和结果格式。
- Agent 的状态与消息历史。
- 多轮工具调用与循环终止条件。

**开发任务**

1. 实现统一的 LLM Provider 接口。
2. 支持至少一种 OpenAI-compatible 模型 API。
3. 实现基础消息结构：System、User、Assistant、Tool。
4. 实现 Agent Loop：
   - 发送模型请求；
   - 解析模型返回；
   - 检测工具调用；
   - 执行工具；
   - 将执行结果加入上下文；
   - 继续模型推理直至任务结束。
5. 支持多个工具调用、最大迭代次数和取消操作。
6. 添加基础流式输出。

**验收标准**

能够通过自然语言触发工具调用，工具执行完成后，模型可以继续推理并给出最终结果。

### Phase 2：Coding Tools 工具系统（核心）

**学习内容**

- Function Calling 的工具定义。
- 工具注册、参数校验、返回值规范。
- 文件系统操作与路径安全。
- subprocess 与 asyncio 子进程调用。

**开发任务**

实现统一的 Tool 接口与 Tool Registry。

基础工具：
- `read_file`：读取指定文件。
- `write_file`：创建或覆盖文件。
- `edit_file`：按照指定范围或内容修改文件。
- `list_files`：浏览目录结构。
- `search_files`：搜索文件名称或文本内容。
- `bash`：执行受控 Shell 命令。

要求：
1. 所有工具使用结构化 Schema。
2. 支持参数验证和统一错误返回。
3. 文件操作限制在授权工作目录。
4. 危险操作必须经过用户确认。
5. 命令执行支持超时、输出截断与退出码。
6. 文件编辑支持修改前检查，避免误覆盖。

**验收标准**

Agent 能够独立完成一个简单编程任务，例如：

“创建一个 Python 模块，实现快速排序，编写测试并执行，发现错误后自动修复。”

能够形成完整的“读文件 → 修改代码 → 执行测试 → 分析错误 → 再次修改”流程。

### Phase 3：会话管理与上下文工程（核心）

**学习内容**

- Context Window 与 Token Budget。
- 多轮对话上下文构造。
- Session 持久化与消息恢复。
- Context Compaction 原理。
- 长任务中的上下文污染和信息丢失问题。

**开发任务**

1. 采用 JSONL 存储会话事件。
2. 保存用户消息、模型回复、工具调用与执行结果。
3. 实现 Session 创建、加载、恢复和切换。
4. 实现上下文 Token 估算。
5. 实现可配置的上下文长度阈值。
6. 实现上下文压缩：
   - 保留最近的关键交互；
   - 对较早历史生成摘要；
   - 保留当前任务目标、关键文件、已完成工作和待解决问题；
   - 避免压缩后丢失工具调用状态。
7. 实现会话中断后的任务恢复。

**进阶功能（可选）**

参考 Pi 实现 Session Tree，通过节点 ID 与 Parent ID 支持会话分支和回溯。

**验收标准**

Agent 可以进行较长时间的多轮编程任务，在退出并重新启动后恢复历史；达到上下文阈值时能够自动压缩，并继续执行任务。

### Phase 4：工程可靠性与安全机制

**学习内容**

- Agent 错误处理与恢复策略。
- Shell 执行风险与权限控制。
- Docker 容器隔离。
- 结构化日志与执行追踪。
- 流式模型请求的异常处理。

**开发任务**

1. 实现工具调用错误捕获和结果回传。
2. 处理模型超时、API 失败与限流。
3. 支持用户中断 Agent 任务。
4. 对危险文件操作和命令执行添加权限确认。
5. 通过 Docker 或其他受限执行环境隔离不可信命令。
6. 增加 Agent 执行日志，记录：
   - 模型请求；
   - 工具名称和参数；
   - 执行结果；
   - Token 消耗；
   - 执行耗时；
   - 异常信息。
7. 为关键模块添加自动化测试。

**验收标准**

Agent 在工具执行失败、模型超时或用户中断时不会直接破坏整个会话；错误能够被定位，危险操作受到限制。

注意：仅限制工作目录或依靠 Prompt 不足以保证 Shell 安全。涉及不可信代码时，应使用真正的执行隔离。

### Phase 5：CLI 与交互体验

**学习内容**

- Python CLI 开发。
- 终端流式渲染。
- 交互式命令与状态展示。

**开发任务**

1. 使用 Typer + Rich 构建交互式终端。
2. 实现模型回答实时输出。
3. 展示工具调用、执行状态和错误信息。
4. 实现基础命令：
   - `/new`：新建会话；
   - `/resume`：恢复会话；
   - `/model`：切换模型；
   - `/compact`：手动压缩上下文；
   - `/status`：查看 Token、会话和任务状态；
   - `/exit`：退出。
5. 支持 Ctrl+C 中断运行。
6. 实现基础配置管理。

**验收标准**

可以通过终端连续完成真实的代码修改与测试任务，而不是只能执行预设 Demo。

### Phase 6：Agent 评测系统（重点）

**学习内容**

- Agent Benchmark 的基本设计。
- Tool Calling 评测方法。
- Task Success Rate、工具调用成功率。
- Token 成本、任务延迟与失败分析。
- 消融实验与可重复性。

**开发任务**

1. 建立 20～50 个小型 Coding Tasks。
2. 覆盖文件检索、代码修改、单元测试、错误修复、多文件修改等场景。
3. 为每个任务设计：
   - 初始项目环境；
   - 自然语言任务描述；
   - 自动化测试或验收规则；
   - 最大执行步数和资源限制。
4. 实现自动化评测 Runner。
5. 每次运行使用独立的、可重置的测试工作区。
6. 记录：
   - Task Success Rate；
   - 工具调用成功率；
   - 平均工具调用次数；
   - Token 消耗；
   - 任务耗时；
   - 失败类型。
7. 对比不同模型、上下文策略与工具配置下的 Agent 表现。
8. 输出评测报告和结果图表。

**验收标准**

拥有能够重复运行的 Agent 评测集，能通过真实数据分析系统效果，而不是凭主观体验判断 Agent 好坏。

---

## 4. 算法扩展：基于 MiniMind 的 Tool Calling 优化

该阶段为独立扩展，不影响 Code Agent 工程主体。

### Phase 7：工具调用轨迹数据集

**学习内容**

- SFT 训练数据格式。
- Tool Calling 的结构化输出。
- Agent 轨迹与模型决策数据。
- 数据清洗、训练集与测试集划分。

**开发任务**

1. 从 Agent 评测任务中收集工具调用轨迹。
2. 提取用户指令、上下文、工具定义和模型生成的工具调用。
3. 构建工具选择与参数生成训练样本。
4. 清洗错误样本并标注正确工具调用。
5. 构建独立的训练集、验证集与测试集。
6. 避免同一任务及其变体泄漏到训练和测试两侧。

**验收标准**

形成可以用于 SFT 的结构化工具调用数据集，以及独立的测试集。

### Phase 8：MiniMind SFT 与模型优化

**学习内容**

- Transformer 与 Causal LM。
- SFT Loss、Label Masking、Chat Template。
- LoRA 微调原理。
- Tool Calling 训练与推理格式。
- 模型评测和消融实验。

**开发任务**

1. 回顾此前 MiniMind 预训练代码与实验。
2. 跑通 MiniMind SFT 流程。
3. 设计与 Agent 工具协议一致的数据格式。
4. 基于工具调用数据开展 SFT 或 LoRA 微调。
5. 对比基础模型与微调模型。
6. 评估：
   - 工具选择准确率；
   - 工具参数正确率；
   - 工具调用格式有效率；
   - 多步任务完成率。
7. 分析数据规模、训练参数与模型表现的关系。
8. 在 Code Agent 中提供实验模型接入能力。

**验收标准**

完成真实的模型微调与对照实验，获得可复现的指标和实验结论。

注意：MiniMind 的能力有限，优先验证结构化工具调用能力，不将其是否能完成复杂代码生成任务作为唯一成功标准。

---

## 5. 推荐项目结构

```text
timocode/
├── agent/
│   ├── loop.py             # Agent Loop
│   ├── state.py            # Agent 状态
│   ├── context.py          # 上下文管理
│   ├── compaction.py       # 上下文压缩
│   └── session.py          # 会话持久化
│
├── llm/
│   ├── base.py             # Provider 抽象
│   ├── provider.py         # API 接入
│   └── schemas.py          # 消息与模型数据结构
│
├── tools/
│   ├── base.py             # Tool 接口
│   ├── registry.py         # 工具注册
│   ├── filesystem.py       # 文件操作
│   ├── search.py           # 文件搜索
│   └── shell.py            # Shell 执行
│
├── security/
│   ├── permissions.py      # 权限与确认
│   └── sandbox.py          # 执行隔离
│
├── cli/
│   └── app.py              # 终端交互
│
├── evals/
│   ├── tasks/              # 测试任务
│   ├── runner.py           # 自动评测
│   └── metrics.py          # 评测指标
│
├── experiments/
│   └── minimind/          # 模型训练与评测
│
├── tests/
├── config/
├── docs/
├── README.md
└── pyproject.toml
```

目录是目标结构，开发初期可以只创建实际需要的模块。

---

## 6. 关键项目里程碑

| 里程碑 | 达成标准 |
|---|---|
| M1：最小 Agent | LLM 能够循环调用工具并完成简单任务 |
| M2：Coding Agent MVP | 可以读取、修改、测试并修复代码 |
| M3：可持续交互 | 支持 Session、上下文压缩和恢复 |
| M4：工程完成版 | 具备权限控制、执行隔离、日志和自动化测试 |
| M5：可评测版 | 拥有任务集、评测 Runner 和实验结果 |
| M6：算法实验版 | MiniMind 完成 Tool Calling SFT 与对照评测 |

**达到 M2 就形成了一个可演示项目；达到 M4～M5，就具备了更充分的工程简历素材；完成 M6 后，可以进一步整理成算法项目。**

---

## 7. 与 Codex 协作的开发规范

开发时遵循以下原则：

1. 不要一次性生成整个项目。
2. 严格按 Phase 顺序推进。
3. 每个 Phase 开始时，先讲解相关概念和架构，再进行编码。
4. 将开发任务拆分为小模块，每次只实现一个明确功能。
5. 优先给出模块接口、数据结构、实现思路和测试方案。
6. 核心模块完成后解释关键代码、设计原因和替代方案。
7. 为每个核心模块编写 pytest 测试。
8. 每个阶段完成后给出功能演示、测试结果和当前限制。
9. 不主动添加计划外功能，涉及重要架构变化时先解释。
10. 使用 Git 保存阶段性成果，维护 README、架构文档和开发记录。
11. 不允许用 Mock 测试通过冒充真实 LLM 端到端测试成功。
12. 任何简历相关的性能或效果指标必须来自实际测量。

---

## 8. 最终交付成果

**工程成果**
- 可运行的 Python Coding Agent。
- 终端交互与多模型接口。
- 文件操作和受控命令执行。
- 会话管理与上下文压缩。
- 安全控制、日志追踪与自动化测试。
- 完整的项目 README 和架构文档。

**评测成果**
- 可重复执行的 Coding Task 数据集。
- 自动化 Agent Evaluation Runner。
- 模型与上下文策略的实验对比。
- 任务成功率、成本和失败分析报告。

**算法成果（可选）**
- Tool Calling 数据集。
- MiniMind SFT 微调代码及模型。
- 基础模型与微调模型的效果对比。
- 消融实验与模型评测报告。

**最终目标：实现一个架构清晰、功能可靠、可独立使用、可量化评测、具备算法扩展能力的 Python Coding Agent。**
---

## 9. 已确认的协作方式与当前进度

### 9.1 协作方式

- 用户熟悉 Python，采用由助手编写代码并讲解核心逻辑的结对开发方式。
- 每次编码前，先讲清相关原理、模块设计、接口、技术取舍和测试方案。
- 每次只推进一个可独立验证的模块，严格遵循 Phase 顺序，并在阶段验收后进入下一阶段。
- 助手实现当前模块及其测试，再解释关键代码、执行路径与设计原因，帮助用户理解并能够复述。
- 阅读 Pi 源码时，使用 Python 概念解释 TypeScript 中的相关设计。
- 每次结束时总结已完成内容、实际测试结果、存在的问题和下一步计划。
- 模拟测试和真实 LLM 端到端测试分别报告，所有效果指标以实际测量为准。

### 9.2 当前进度

- 当前处于 Phase 0，已完成整体架构导读、任务拆分讨论和 0.1 环境与测试基线。
- 已初始化 Python 项目和 Git 仓库，并在独立虚拟环境完成可编辑安装。
- 安装与导入冒烟测试 1 项通过，pip check 通过，Git 忽略规则已验证。
- 尚未阅读并核对 Pi 官方架构文档和核心源码。

### 9.3 Phase 0 的后续小任务

1. 0.1 环境与测试基线。检查 Python 与 Git，初始化最小项目、虚拟环境、依赖配置、环境变量示例与 pytest，验证安装和导入。
2. 0.2 消息与工具调用导读。通过完整交互时序理解消息角色、工具定义、工具调用及结果回传。
3. 0.3 Pi 核心路径阅读。按 packages/ai、packages/agent、packages/coding-agent 的顺序，理解模型接口、Agent Loop 与应用层组织。
4. 0.4 架构与接口定稿。明确 Agent、LLM Provider、Tool、Session、Context 的职责与接口，完成 Phase 0 验收。

### 9.4 任务 0.1 的范围与验收

任务 0.1 已完成环境与测试基线。

实际使用 Python 3.11.5、venv、pip、setuptools、pyproject.toml、src/timocode 包布局和 pytest 8.4.2。运行时依赖为空，pytest 通过 dev 可选依赖安装。

验收要求为项目可在独立虚拟环境安装、timocode 可导入、python -m pytest 通过，并正确忽略虚拟环境和本地密钥配置。冒烟测试仅验证安装与测试链路。

下一项为 0.2 消息与工具调用导读，通过具体交互时序解释模型请求、工具调用与结果回传。Provider 和 Agent Loop 的实现安排在 Phase 1。

### 9.5 项目命名约定

- 项目展示名称为 TimoCode，英文标识为 timocode。
- CLI 命令为 timo。
- Python 包名采用 timocode。
- 命名灵感参考 KiloCode，架构学习继续按计划参考 Pi。

### 9.6 开发记录

#### 2026-11-07，Phase 0.1

- 创建 pyproject.toml、src/timocode/__init__.py、tests/test_package.py、.gitignore、.env.example 和 README.md。
- 使用现有 Python 3.11.5 创建 .venv，完成 pip install -e ".[dev]"。
- 测试从临时目录启动隔离模式 Python，验证安装后的包导入及分发元数据读取。
- python -m pytest -v 的结果为 1 passed，pip check 的结果为 No broken requirements found。
- 验证虚拟环境、本地环境变量、Python 缓存和构建元数据被 Git 忽略，环境变量示例文件可跟踪。
- 当前环境变量模板尚待 Phase 1 实现读取，CLI 命令在 Phase 5 实现，Pi 源码阅读尚待完成。
