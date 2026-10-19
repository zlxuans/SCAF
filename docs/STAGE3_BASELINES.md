# 阶段 3：官方基线和统一 RGB-D 里程计

阶段 3 只在 train/val 上建立开发基线。算法增强、训练、真实低照度最终测试和 GPU 性能基准属于后续阶段。评价定义见 [实验协议 v2](EXPERIMENT_PROTOCOL.md)。

## 固定资产

`configs/baselines-v1.yaml` 固定以下来源：

- [LightGlue 官方仓库](https://github.com/cvg/LightGlue)，提交 `eb42fee2d71449efb0aa5c10549752b5d75384d8`，使用其中的 SuperPoint 和 LightGlue 实现。
- [FeatureBooster 官方仓库](https://github.com/SJTU-ViSYS/FeatureBooster)，提交 `0260868806d0ae603a166848d64078bfd1c2f1f9`，使用 SuperPoint+Boost-F。
- SuperPoint 权重 SHA-256：`52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e`。
- SuperPoint LightGlue 权重 SHA-256：`6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a`。
- FeatureBooster 权重 SHA-256：`5334d9aa861e877a2b99baff0d682e1ac8a749cdd65eb1d4b8bd0a8bb8bf0359`。

这些权重哈希是本次下载后的版本标识，不是发布方签名。准备入口核对仓库提交、受版本控制文件未修改和权重哈希，并记录相关源码文件哈希。评测开始前重新核验全部资产，不允许静默更新或随机权重回退。原始源码和权重保留服务器，不复制进公开仓库；使用条件遵循各官方仓库，SuperPoint 的许可与 LightGlue 本体不同。

SuperPoint 直接调用官方 forward，禁用自动缩放，OpenCV 灰度输入，不修改检测器。FeatureBooster 输入是官方约定的归一化位置及置信度，输出保持 256 维；所有方法共享同一份 512 点预算特征。FeatureBooster 演示脚本使用更高的检测阈值，本项目统一用 0.0005，属于统一条件的比较而不是复刻其论文分数。

适配器只加载 LightGlue 所需的三个官方模块，绕过导入其他检测器的包入口，避免为未使用的模型增加 torchvision 等依赖。官方文件保持原样。SuperPoint 构造器的下载回调被本地已核验 state_dict 替代。LightGlue 按官方规则迁移旧注意力层名，允许缺少确定性计算的 `confidence_thresholds` buffer，以及九个不再使用的 `log_assignment.i.r` 标量；其他缺失或额外键报错。兼容项保存到运行记录。

## 服务器准备

以下命令均在服务器 `/developer06/SCAF` 执行，默认 CPU，不修改其他项目环境：

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pip install -r requirements/stage3.txt
.venv/bin/python -m scaf.baselines.prepare
```

若环境没有 pip，先用该独立解释器运行 `-m ensurepip`。准备入口是唯一允许下载的步骤；评测使用本地源码、权重和锁清单。已有清单只允许完全一致地复用，版本或文件改变须建立新的资产目录与清单。

## 检查和开发运行

```bash
export SCAF_AUDIT_MANIFEST=/developer06/SCAF/runs/stage2-20261019T102938-1156259/audit/manifest.json
export SCAF_BASELINE_ASSET_MANIFEST=/developer06/SCAF/weights/baselines/manifest.json
bash scripts/check_stage3.sh
```

使用独立 worktree 时，遵守服务器说明中的固定提交方法，并明确设置：

```bash
export SCAF_PYTHON=/developer06/SCAF/.venv/bin/python
export SCAF_DATA_ROOT=/developer06/SCAF/data
export SCAF_ASSET_MANIFEST=/developer06/SCAF/weights/baselines/manifest.json
# 必须使用新的绝对目录，检查脚本及其 evaluation 子目录拒绝覆盖已有结果。
export SCAF_STAGE3_OUTPUT=/developer06/SCAF/runs/stage3-NEW
bash scripts/check_stage3.sh
```

默认取三条开发序列的前 120 个按步长 3 选中的帧，分别运行 clean、medium。运行前重新核验所用开发序列的全部文件哈希。若要完整开发序列：

```bash
$SCAF_PYTHON -m scaf.evaluation.cli --data-root /developer06/SCAF/data \
  --audit-manifest "$SCAF_AUDIT_MANIFEST" --asset-manifest "$SCAF_ASSET_MANIFEST" \
  --max-frames 0 --output /developer06/SCAF/runs/stage3-full-NEW
```

评测要求工作区干净且从当前源码根目录运行。测试角色在提取特征之前拒绝使用；ETH3D 保留测试集继续只做阶段 2 文件与标定审计。

## 输出与失败处理

每次输出保留 `manifest.json`、`dependencies.txt`、汇总 JSON/CSV、每个条件的选帧清单和逐帧索引。每帧共享特征 NPZ 包含位置、描述子、分数、时间戳、图像尺寸和原序列索引；FeatureBooster 单独存增强描述子。每对匹配 NPZ 保存匹配索引、分数、独立几何标签、重投影误差、PnP 内点和相对位姿。

`pairs.jsonl` 包含每对成功/失败状态、原因、匹配质量和分项计时。每段 TUM 文件保留自己的单位起点坐标系；`trajectory.npz` 保留完整 SE(3)、时间戳和段编号。失败不会输出伪位姿或借助真值重置；所有中断都会留在分母中。

完整 ATE 只在全部选中帧间估计成功、至少三个关联真值位置时报告。发生中断则只保留各段 ATE，完整值为 null。RPE 只计算同段相邻选中帧且两端真值均有效的相对变换，不跨参考缺失或跟踪失败。位置对齐退化的轨迹记录位置秩；不能把低秩对齐当成充分的旋转约束。

初始化与完整性核验耗时单列，尚不包括 Python 解释器与模块导入冷启动。计时跳过最初 10 对，CPU 分项 P50/P95 是共享主机上的开发记录，不能用来支持 GPU 实时性结论。作业 wall time 包含三方法调度、独立评价与结果写盘，不能等同于某一前端的延迟。
