# 服务器运行说明

所有代码运行和测试均在服务器 `/developer06/SCAF` 执行，本地仅编辑与同步。

## 双卡共享规则

服务器有两张 TITAN RTX 24 GiB，阶段 1 检查时两张卡均有其他任务。默认只使用 CPU，线程数为 1。不要终止其他任务、切换显卡模式、修改系统 CUDA、更新其他项目的环境或自动启动双卡训练。

```bash
cd /developer06/SCAF
.venv/bin/python -m scaf.cli doctor
bash scripts/check.sh
```

CUDA 诊断必须显式指定物理卡索引，通过 nvidia-smi 查询后转换为 UUID 并限制可见设备。默认检测到显存使用超过 256 MiB 或利用率超过 10% 即拒绝；检测和锁不能阻止其他项目抢占，因此不构成调度器。

```bash
# 默认仅在指定卡空闲时运行
.venv/bin/python -m scaf.cli smoke --device cuda --gpu 0

# 仅允许当前微型诊断在共享 GPU 上运行：32×32 矩阵，默认 3 步
.venv/bin/python -m scaf.cli smoke --device cuda --gpu 0 --allow-busy-gpu
```

共享诊断仍要求至少 1 GiB 空闲，PyTorch 分配器上限为设备显存的 5%；CUDA 上下文和外部库显存不受该分配器上限完全约束。它会短时占用计算资源，不应当作训练或效率实验。实际训练资源分配在开始训练阶段前重新检查，不直接沿用本次 GPU 选择。

## 代码同步与固定提交

本地提交后推送 `origin` 和 `server`。服务器主目录启用了 `receive.denyCurrentBranch=updateInstead`，仅在工作区干净时接受更新。不要强推或覆盖服务器的未提交改动。

运行期间使用主目录以外的 detached Git worktree，避免下一次推送修改正在执行的源码：

```bash
cd /developer06/SCAF
revision=$(git rev-parse HEAD)
bash scripts/snapshot.sh "/developer06/SCAF-worktrees/$revision" "$revision"
cd "/developer06/SCAF-worktrees/$revision"

# 必须指向快照源码，避免共享环境中的 editable 安装读到主目录
export PYTHONPATH="$PWD/src"
export SCAF_PYTHON=/developer06/SCAF/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$SCAF_PYTHON" -m scaf.cli smoke --device cpu --output-root /developer06/SCAF/runs
```

`snapshot.sh` 要求主目录干净、目标路径为目录外的新绝对路径；已存在快照可直接复用。阶段 1 的测试会提交主目录新版本，验证 detached 快照文件及提交未改变。不要在快照中执行开发修改或更新依赖；后续依赖变更也需要为旧任务保留原环境。

## 后台启动与状态

`launch_smoke.sh` 只启动基础设施诊断，当前还没有训练任务：

```bash
# 返回一个 launcher 目录，其中有 pid 和 console.log
SCAF_PYTHON=/developer06/SCAF/.venv/bin/python bash scripts/launch_smoke.sh \
  --device cpu --output-root /developer06/SCAF/runs

# 将 JOB_DIR 替换为实际返回目录；使用脚本实际返回路径
cat JOB_DIR/pid
tail -n 30 JOB_DIR/console.log
```

PID 只作辅助定位，进程号可能复用；最终以 `manifest.json` 的 completed/failed 状态为准。崩溃或节点退出导致 running 状态未更新时，先查明对应进程是否存在，再决定恢复，勿将其他项目 PID 当成本项目任务。

## 诊断恢复

在相同干净提交中，将 smoke 的总步数增加并指向已保存检查点：

```bash
python -m scaf.cli smoke --device cpu --steps 1
# 使用上一命令输出目录中的 checkpoint.pt
python -m scaf.cli smoke --device cpu --steps 3 --resume RUN_DIR/checkpoint.pt
```

每次恢复创建新目录，不覆盖原日志。损坏/错误版本/配置不一致的检查点会报错；训练模型的断点恢复是阶段 6 的独立工作。本阶段不调用 GPU 长循环，不开始数据下载或正式训练。

CUDA 确定性检查在导入 PyTorch 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；该设置只影响本次 SCAF 进程，不修改其他训练进程。
