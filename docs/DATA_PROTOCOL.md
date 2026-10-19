# 数据与监督协议 v1

执行位置为服务器 `/developer06/SCAF`。数据和生成物不上传 GitHub；仓库提交可复现的配置、代码和检查摘要。

## 数据身份、来源与划分

冻结清单为 `configs/datasets-v1.yaml`，完整序列参与划分，不拆相邻帧。相同场景身份、相同序列 ID、目录别名及包含关系会被检查拒绝。补充数据或改变角色时建立新版本，并重新做泄漏审查；不能修改旧运行所引用的清单。

| 序列 | 角色 | 场景组 | 用途 |
| --- | --- | --- | --- |
| TUM fr1_xyz | train | tum_fr1_small_office | 管线开发及初步学习 |
| TUM fr2_desk | val | tum_office_conservative_group | 正常场景验证 |
| TUM fr3_walking_xyz | val | tum_office_conservative_group | 动态干扰验证 |
| ETH3D einstein_flashlight | test | eth3d_einstein | 保留的真实手电筒照明测试 |

Freiburg 1 是小办公室录制组。对于 fr2/fr3，不用不同前缀推断场景独立，保守合并到同一验证组。未来 ETH3D einstein 的其他照明变体必须与 flashlight 同属测试角色。这里的 test 是本研究自定角色：该 ETH3D 序列属于官方公开真值的 training 子集。不能把它称为官方隐藏测试集。

这是管线起步版本：训练仅一个短序列，独立真实低照度仅一个场景，不足以直接支撑论文泛化结论。阶段 6 正式训练前评估训练场景规模；阶段 7 的目标仍是至少两个独立真实低照度序列。新增数据先明确场景身份再下载/分配。当前 test 禁止生成匹配、监督预览和轨迹分数，只做文件/标定完整性检查；不能据其结果选阈值。

来源及使用条件于 2026-10-19 核查：

- [TUM 官方主页](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)：当前数据许可 CC BY 4.0，引用 Sturm 等 IROS 2012 的数据集论文。
- [TUM 下载与场景描述](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)。
- [ETH3D 官方主页](https://www.eth3d.net/)：数据许可 CC BY-NC-SA 4.0；本项目用于非商业研究，原始图像和派生可视化均留在本地/服务器，不纳入公开代码提交。
- [ETH3D 下载页](https://www.eth3d.net/slam_datasets)及[SLAM 数据说明](https://www.eth3d.net/slam_documentation)，论文引用 Schöps 等 CVPR 2019《BAD SLAM: Bundle Adjusted Direct RGB-D SLAM》。

本次从旧项目复用的是五个原始下载压缩包，未继承旧划分、模型或结果。配置的 SHA-256 是本次本地实测后在服务器重新计算核对的版本标识，**并非发布方签名或官方校验值**。来源 URL 是核查得到的官方分发地址；本次没有重新下载全量文件。解压时检查路径越界、链接、重复文件和压缩包哈希，ZIP 读取同时验证 CRC；目标已存在时拒绝覆盖。每次数据审计进一步记录所有解压文件的 SHA-256、字节数，并解码全部 PNG。来源可追溯不等于验证过官方密码学真实性。

## 图像、深度与位姿约定

根据 [TUM 格式说明](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)和 ETH3D SLAM 说明，两者发布的深度均已对齐到对应图像视点，uint16 PNG 除以 5000 得米，0 表示缺失。不再次乘 TUM 已应用的深度校正因子。加载器还将 0.2—8 m 外深度置 0；这是当前前端使用范围，不改原文件。

TUM v1 使用官方建议的 ROS 默认内参 `(525,525,319.5,239.5)`，不单独去畸变、不对已经配准的深度再投影。该选择是明确记录的近似，不是逐设备精确标定。所有方法共用，边缘、快速运动及 RGB/Depth 时间差仍可能导致几何标签误差。若后续改标定，需升版、重新审查投影并重新生成标签。ETH3D 使用原包 `calibration.txt` 的四个参数，其发布图像已去畸变；不套用 TUM 内参。

像素 `(0,0)` 是左上像素中心，相机轴为 x 右、y 下、z 前。输入最大 640×480，保持比例、不放大；图像 area 缩小、深度最近邻缩小。内参按实际尺寸更新 `f'=sf, c'=s(c+0.5)-0.5`，不对无效深度插值。

时间戳必须严格递增且有限；重复索引文件、越界文件路径和缺失文件直接报错。RGB/Depth 在 20 ms 内按全局最近优先进行一对一关联，不复用深度帧；未关联帧数显式报告。ETH3D 要求配对时间戳完全相同。

真值记录格式为 `t tx ty tz qx qy qz qw`，平移单位米；矩阵是 **T_world_camera**。相邻真值间平移线性插值，旋转用最短弧 SLERP；只在间隔不超过 50 ms 的已知两侧之间插值，不外推。真值缺失/间断的帧仍可加载，但不生成几何监督，不计算需要真值的轨迹分数。

## 几何监督

对源关键点采样干净深度，反投影后使用 `T_camera1_camera0 = inverse(T_world_camera1) @ T_world_camera0` 投向目标相机。检查源深度、有限数、正 z、目标图像边界，以及投影位置和候选关键点位置的目标深度。两处深度均须满足 `abs(d_target-z_projected) <= 0.05 m + 0.02*abs(z_projected)`。这同时排除部分遮挡、深度不一致和动态物体残差，但**不能证明剩余点是静态点**。

候选关键点投影距离不超过 3 px；以距离、源序号、目标序号排序做确定性一对一贪心分配，解决重复目标。`matches0>=0` 是保留正对应，`-1` 是忽略/未知，**不能直接作为负样本**。`reason` 保留拒绝原因，投影坐标、深度和相对变换随 NPZ 保存。后续区分性损失要另行定义可信负样本规则。

本阶段用 16 px 网格做投影 QA，默认间隔 3 个已关联帧，序列首/中/尾各抽一对。这些数据是接口与几何检查，正对应数不是描述子匹配分数，也不能当作论文结果。阶段 3 的检测器关键点可以直接使用同一监督函数。

## 退化与可复现性

配置 `configs/degradations-v1.yaml` 预定义 clean/low/medium/severe、深度缺失、深度噪声及组合设置。RGB 执行 `gain*(I/255)^gamma + N(0,std)` 后裁剪量化；这是强度域合成退化，不是物理相机成像或真实夜间替代。深度噪声标准差为 `absolute_std + relative_std*depth`，再做独立像素随机缺失；原本无效的像素保持无效，不会由噪声补出深度。

种子 13 与训练种子独立。以配置变体名、序列/帧身份、退化种子计算稳定 SHA-256 种子，RGB 和深度各用独立随机流，结果不依赖进程 hash、加载顺序或 worker 数。改变退化配置必须随运行记录保留完整参数。干净 RGB/Depth 不原地修改：几何标签来自干净参考；阶段 7 另行注明受扰深度是否也输入里程计后端。clean 与真实 flashlight 的角色在清单中明确区分。

## 服务器运行

先把原始压缩包放入 `data/downloads/`。然后在项目根目录执行（均为 CPU，不初始化 CUDA）：

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# 分别对清单中的四个 ID 执行，已存在的目录不重复解压
.venv/bin/python -m scaf.data.cli extract --sequence tum_fr1_xyz
# 唯一输出目录，拒绝覆盖旧审计或预览
nice -n 10 ionice -c 3 .venv/bin/python -m scaf.data.cli audit --output runs/data-audit-NEW
.venv/bin/python -m scaf.data.cli preview --sequence tum_fr1_xyz \
  --audit-manifest runs/data-audit-NEW/manifest.json --output runs/data-qa-NEW
```

审计输出 `files.jsonl`（全部文件哈希）、`summary.json`（数量/时间/标定/真值覆盖）和 `manifest.json`（代码提交、dirty 状态、依赖、清单/结果哈希、命令、完成或失败状态）。预览需传入同版本的已完成审计，并重新核对所选序列文件；数据变化则拒绝继续。输出中断也保留失败记录。正式运行应从干净提交启动，修改源码期间不运行任务。

运行完整检查与三条开发序列 QA：`bash scripts/check_stage2.sh`。每次创建唯一运行目录，耗时主要在解码和哈希；不启动训练。阶段检查结果见 `STAGE2_REPORT.md`。
