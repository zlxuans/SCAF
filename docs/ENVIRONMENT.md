# 环境安装与目录约定

## 独立环境

参考环境为 Python 3.12、NumPy 1.26.4、PyYAML 6.0.2、PyTorch 2.5.1。本地安装 CPU 版 PyTorch，服务器使用 CUDA 12.1 版，避免触碰已有 RL 环境或系统 CUDA。基础依赖在 `requirements/`，每次实际运行另保存完整已安装依赖快照。

```bash
# 在项目根目录，使用可用的 Python 3.12 创建独立虚拟环境
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.txt

# 本地 CPU；服务器改用下方 cu121 索引，二选一
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
# .venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
bash scripts/check.sh
```

若系统没有 Python 3.12，可以使用 uv 管理独立解释器：`UV_PYTHON_INSTALL_DIR=$PWD/.python uv venv --python 3.12 .venv`，再将上述 pip 命令替换为 `uv pip install --python .venv/bin/python ...`。服务器已经准备好 `.venv` 与 `.python`，无需重复安装。

PyTorch 对应版本的 CPU/cu121 安装命令来源：[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。Python 包中的 CUDA 运行时与系统编译工具链不同，本阶段不编译 CUDA 扩展。

已验收环境的完整版本列表在 `requirements/lock-cpu-py312.txt` 和 `requirements/lock-cuda121-py312.txt`。复建时先用对应 PyTorch 索引安装固定 torch 版本，再通过默认 Python 包索引安装该列表的其他依赖（torch 已安装时无需重新解析其来源），最后安装本项目。CPU/CUDA 锁文件分开维护，不互相覆盖。

若 shell 设置了其他项目的 PYTHONPATH，运行前使用 `export PYTHONPATH="$PWD/src"`，检查与后台脚本已自动设置；依赖快照仅枚举本虚拟环境，避免把系统 ROS 包误写入安装清单。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/scaf/` | 配置、运行记录、检查入口；后续增加算法模块 |
| `configs/` | 运行配置和实验协议 |
| `requirements/` | 固定基础依赖 |
| `scripts/` | 检查、独立工作目录和后台启动入口 |
| `tests/` | 配置、资源选择、恢复及版本隔离测试 |
| `docs/` | 公开协议和进度，原始 DOCX 仍只在本地 |
| `data/` | 本地数据挂载/索引，Git 忽略；阶段 2 再确定实际数据盘 |
| `runs/` | 独立运行目录、日志与诊断检查点，Git 忽略 |

`doctor` 仅做版本和 nvidia-smi 清单读取，不初始化 CUDA：

```bash
.venv/bin/python -m scaf.cli doctor
.venv/bin/python -m scaf.cli doctor --output runs/environment.json
```

不要将运行环境的完整进程命令行、环境变量列表或凭据写入公开文档。
