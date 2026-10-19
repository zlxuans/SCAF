# 阶段 1 验收记录

日期：2026-10-19（Asia/Shanghai）。运行与测试目标：服务器 `/developer06/SCAF`。本地只负责编辑、审阅和 Git 同步。

## 交付内容

- 可安装的 `src/scaf` 包；`scaf doctor` 只读盘点环境，`scaf smoke` 运行有限步张量检查。
- 严格的 YAML 配置校验、训练/退化种子分离、唯一输出目录、配置/环境/提交记录、依赖快照和失败状态。
- 指定物理卡后按 GPU UUID 限制可见设备；默认拒绝共享卡，诊断锁避免两个 SCAF 检查争抢同一卡，空闲显存不足 1 GiB 时始终拒绝。
- 诊断任务的 RNG/步骤检查点恢复、后台日志、固定提交 detached worktree，以及实际源码目录校验。
- 实验协议 v1、环境说明、服务器操作说明、CPU/CUDA 完整依赖版本快照。

## 环境盘点

| 项目 | 服务器记录 |
| --- | --- |
| GPU | 两张 NVIDIA TITAN RTX，各 24,576 MiB |
| 驱动 | 550.100 |
| 既有任务 | 阶段开始时两卡分别使用约 9,676 / 15,896 MiB；两个 RL 训练进程持续运行 |
| SCAF 解释器 | Python 3.12.15，项目下独立 `.python` 与 `.venv` |
| PyTorch | 2.5.1+cu121，独立安装；不升级 RL 环境或系统 CUDA |
| 基础依赖 | NumPy 1.26.4、PyYAML 6.0.2、pytest 8.3.5 |
| 项目数据盘 | `/developer06` 挂载盘，检查时约 47 TiB 可用，阶段 2 再核定数据位置 |
| CPU | 本阶段按 1 线程运行，不启动数据加载工作池 |

最初在旧 `/root/SCAF` 进行的准备记录保留为历史；当前同步远端已改为 `/developer06/SCAF`，正式阶段验收使用新目录。旧目录和既有 RL 环境均保留，不进行清理。

## 检查结果

验收代码提交：`1ce038ffc7d5fdec08db28dbe62d565a4961e1a2`。后续仅补全文档，未修改验收代码。

| 检查 | 结果与记录 |
| --- | --- |
| 服务器自动测试 | `bash scripts/check.sh`：8 passed，14.73 秒；覆盖配置校验、CPU 隔离、忙卡拒绝、GPU UUID、同卡互斥、显存底线、恢复及快照隔离 |
| CPU 诊断 | `runs/stage1_smoke_20261019T095743_6ba3aa41`，completed，工作区干净 |
| GPU 0 诊断 | `runs/stage1_smoke_20261019T095745_273b0c7e`，completed，显式允许微型共享检查 |
| PyTorch GPU 峰值分配 | 33,566,720 字节，约 32.01 MiB；这是分配器记录，不包含全部 CUDA 上下文占用 |
| 忙 GPU 1 默认拒绝 | `runs/stage1_smoke_20261019T095854_9691259f`，failed，保留明确 busy 错误，无 CUDA 张量计算 |
| 后台入口 | 快照 `runs/launcher/job_ubxcvETC/console.log` 指向 `runs/stage1_smoke_20261019T095854_38c89abd`，completed |
| 检查点恢复 | `runs/stage1_smoke_20261019T095924_334ab1cf`，与不间断运行 `runs/stage1_smoke_20261019T095926_07e00e66` 三步校验值完全相同 |
| 固定提交目录 | `/developer06/SCAF-worktrees/stage1-1ce038f`，detached HEAD，实际源码路径指向该目录 |
| 既有训练 | 两个原 RL 训练进程仍在运行；诊断退出后两卡显存回到此前约 9,676 / 15,896 MiB 的占用 |

上表运行目录均位于 `/developer06/SCAF`，后台 launcher 目录则位于注明的固定提交目录。完整环境和运行日志存放在服务器 `runs/`，不提交 Git。

首次在旧目录的 GPU 检查曾因 cuBLAS 确定性配置缺失失败；已在导入 PyTorch 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并在新目录重新验收通过。这个设置及有效的 CUDA_VISIBLE_DEVICES 均写入运行记录。

阶段 1 的所有任务已完成。现有 GPU 正被其他任务使用，因此没有启动正式训练或速度基准；共享微型检查只能证明代码和 CUDA 可运行，不能证明资源独占或测得公平性能。

## 范围与下一阶段

该阶段只验证基础设施：未训练模型，未运行数据集实验，也没有算法精度或实时性结论。后台入口只支持微型诊断；完整模型/优化器恢复、训练调度与资源规划在后续阶段实现。

下一阶段为阶段 2：数据清单、场景隔离划分、标定/深度检查和监督生成。开始前重新检查 GPU 占用；GPU 训练和正式计时不使用共享诊断的覆盖选项。
