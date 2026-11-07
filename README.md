# TimoCode

使用 Python 从零构建轻量级终端 Coding Agent，架构学习参考 [Pi Coding Agent](https://github.com/badlogic/pi-mono)。项目标识和 Python 包名为 `timocode`，计划中的 CLI 命令为 `timo`。

## 当前进度

Phase 0.1，环境与测试基线。当前包提供最小安装入口，CLI 在 Phase 5 实现。

完整路线及协作约定见 [PLAN.md](PLAN.md)。

## 环境准备

需要 Python 3.11+ 和 Git。在项目根目录的 PowerShell 中执行以下命令。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

直接使用虚拟环境中的解释器，可以明确安装和运行使用的是同一个环境。

`-e` 表示可编辑安装，修改 `src/timocode` 下的 Python 源码后即可使用新代码。`[dev]` 同时安装开发依赖 pytest。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

当前冒烟测试从临时目录启动隔离模式的 Python 子进程，验证 `timocode` 可以导入，并且安装元数据可读取。

## 环境变量

`.env.example` 记录未来模型接入所需的变量占位。接入模型时可复制为 `.env` 并填写本地配置。配置读取将在 Phase 1 实现。`.env` 和 `.venv` 已列入 Git 忽略规则。

## 项目结构

```text
src/timocode/__init__.py  Python 包入口
tests/test_package.py    安装与导入冒烟测试
pyproject.toml           构建、依赖和测试配置
.env.example             环境变量模板
PLAN.md                  开发计划与协作约定
```
