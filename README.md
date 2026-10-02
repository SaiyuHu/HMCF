# HMCF-SAM

面向医学与遥感图像分割的 SAM 微调框架。在 Meta 的 Segment Anything (SAM) 基础上引入 **LoRA 轻量微调**、**HMCF 曲率流掩码精修模块** 与 **TD 边界长度损失**，在训练参数极少的前提下提升分割边界的精度与光滑性。

## 方法概览

| 组件 | 位置 | 说明 |
| --- | --- | --- |
| LoRA 微调 | `segment_anything/lora.py` | 冻结 ViT-B 图像编码器，仅在注意力模块的 q/v 上注入 rank=4 的低秩矩阵 `B·A` |
| HMCF 掩码精修 | `segment_anything/modeling/hmcf_block.py` | `MeanCurvatureFlow`：把 MaskDecoder 输出的 logits 视为水平集函数，用平均曲率流迭代求解（外层 4 次 × 内层 3 次，时间步 0.1，梯度检查点开启）精修边界 |
| TD 边界长度 | `segment_anything/TD.py` | `TDBoundaryLength`：用高斯核卷积快速估计边界长度；`TDLoss`：约束预测与 GT 的边界长度相对差异 |
| 模型组装 | `segment_anything/modeling/SAM_hmcf.py` | `SAM_HMCF`：图像编码器 + 提示编码器 + 掩码解码器 + 可选 HMCF 模块 |
| 模型构建 | `build_sam_hmcf.py` | `samhmcf_model_registry['vit_b']`，支持加载 SAM 权重与已训练的解码器参数 |

**训练损失**：`Loss = BCEWithLogits + 0.1 × TDLoss`。

HMCF 中的曲率参数 `b` 由当前预测的 TD 边界长度与目标边界长度自适应确定：

```
b = max|o| · relu(TD(pred) − TD_target) / (TD(pred) + TD_target + ε)
```

其中 `TD_target` 取训练集掩码 TD 边界长度的 75% 分位数（自动计算并缓存）。

## 三种训练模式

通过 `--model_name` 切换：

| `--model_name` | 编码器 | 损失 | HMCF 精修 |
| --- | --- | --- | --- |
| `SAM` | LoRA | BCE | 否 |
| `SAMtd` | LoRA | BCE + 0.1×TDLoss | 否 |
| `SAMhmcf` | LoRA | BCE + 0.1×TDLoss | 是 |

## 目录结构

```
hmcf-sam/
├── sam_train_with_lora.py          训练入口
├── eval.py                         评测入口
├── build_sam_hmcf.py               SAM-HMCF 模型构建与注册表
├── segment_anything/               SAM 主干及扩展
│   ├── modeling/
│   │   ├── SAM_hmcf.py             SAM_HMCF 整体模型
│   │   ├── hmcf_block.py           MeanCurvatureFlow 掩码精修模块
│   │   ├── image_encoder.py        ViT 图像编码器
│   │   ├── prompt_encoder.py       提示编码器
│   │   ├── mask_decoder.py         掩码解码器
│   │   └── transformer.py          TwoWayTransformer
│   ├── lora.py                     LoRA_sam / LoRA_qkv
│   ├── TD.py                       TDBoundaryLength / TDLoss
│   ├── dice_loss.py                DiceLoss
│   └── utils/                      预处理与 ONNX 导出工具
├── tool/                           数据与指标工具
│   ├── MyDataset.py                数据集与预处理（缩放至 1024、归一化、填充）
│   ├── calculate_metric.py         精度 / 边界光滑性 / 效率指标
│   ├── Membership_degree_map.py    隶属度图生成
│   ├── curvature_map_produce.py    曲率图生成
│   ├── split_busi_dataset.py       BUSI 数据集划分
│   └── resize&split/               各数据集划分、裁剪与加噪脚本
└── cache/td_statistics/            TD 边界长度 75% 分位数缓存
```

## 环境依赖

Python 3.x + PyTorch（建议 CUDA 版本）：

```
torch  torchvision  numpy  opencv-python  Pillow  tqdm  tensorboard  scikit-image  scipy  pyyaml
```

GPU 混合精度训练使用 `torch.amp`，需 PyTorch 2.0 及以上。
`tool/resize&split/chenpengjie.py`（DICOM 读取）额外需要 `SimpleITK` 与 `matplotlib`。

## 数据准备

数据按如下结构放置：

```
dataset/
└── <dataname>/
    ├── train/
    │   ├── image/
    │   └── mask/
    ├── val/
    │   ├── image/
    │   └── mask/
    └── test/
        ├── image/
        └── mask/
```

- `image` 与 `mask` 目录下的文件**按文件名排序后一一对应**，请保证两者文件数量与命名一致。
- 掩码为单通道灰度图，读取后按 `>0` 二值化为前景。

已内置配置的 `--dataname`（见 `sam_train_with_lora.py` 中的 `CONFIG`）：
`ISIC`、`Refuge`、`Kvasir`、`WHU`、`BUSI`、`Kvasir_noisy_std50`、`Kvasir_noisy_std100`、`small`。

其中 batch size 均为 1，学习率均为 1e-4；除 `small` 为 10 个 epoch 外，其余均为 30 个 epoch。

## 预训练权重

将权重放入 `parameterfault/`：

| 文件 | 说明 |
| --- | --- |
| `sam_vit_b_01ec64.pth` | 官方 SAM ViT-B 权重，训练与评测均必需 |
| `medsam_vit_b.pth` | MedSAM ViT-B 权重（可选，用于医学图像对照实验） |

官方 SAM 权重可从 <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth> 下载。

## 训练

```bash
python sam_train_with_lora.py --model_name SAMhmcf --dataname Kvasir --gpu 0
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_name` | `SAMhmcf` | `SAM` / `SAMtd` / `SAMhmcf` |
| `--dataname` | `small` | 数据集名称，决定 batch size / epochs / lr |
| `--sam_checkpoint` | `parameterfault/sam_vit_b_01ec64.pth` | 预训练权重路径 |
| `--gpu` | `0` | 指定使用的 GPU 编号 |
| `--weight_decay` | `1e-8` | 权重衰减 |

输出保存在 `work_dir/<model_name>_runs/<dataname>/<时间戳>/`：

- `lora_best_parameters.pth` / `lora_latest_parameters.pth` —— LoRA 参数
- `decoder_best_parameters/` / `decoder_latest_parameters/` —— 掩码解码器（及 HMCF）参数
- `training.log`、`logs/`（TensorBoard）

说明：随机种子固定为 0；优化器为 Adam，学习率采用带 4 步 warm-up 的余弦退火；当 `--model_name` 为 `SAMhmcf` 时，首次运行会统计训练集掩码的 TD 边界长度 75% 分位数并缓存到 `cache/td_statistics/<dataname>_td_75th_percentile.json`。

## 评测

```bash
python eval.py --model_name SAMhmcf --data_name Kvasir \
    --checkpoint_folder work_dir/SAMhmcf_runs/Kvasir/<时间戳> --test_folder test
```

| 参数 | 说明 |
| --- | --- |
| `--model_name` | `SAM` / `SAMtd` / `SAMhmcf` |
| `--data_name` | 数据集名称 |
| `--checkpoint_folder` | 训练输出目录（内部读取 `decoder_best_parameters` 与 `lora_best_parameters.pth`） |
| `--test_folder` | `test` / `val` / `train` |

结果写入 `test_result/<model_name>/<data_name>/`，包含二值预测掩码 `masks/` 与汇总文件 `evaluation_summary.txt`。

评测指标分为三组：

- **分割精度**：IoU、Dice、Sensitivity、Specificity、Accuracy、Precision、ASSD、Hausdorff 距离、HD95、Betti 误差、紧凑度
- **边界光滑性**：曲率方差、边界粗糙度、分形维度、曲率范围，以及综合光滑度评分
- **分割效率**：平均推理时间、FPS、总参数量、可训练参数量、模型大小、估计 FLOPs / GFLOPs

同时给出效率-精度平衡指标 `FPS × Dice`。

## 工具脚本

| 脚本 | 用途 |
| --- | --- |
| `tool/MyDataset.py` | 数据集类：缩放最长边至 1024、按 SAM 均值方差归一化、右下填充至 1024×1024 |
| `tool/calculate_metric.py` | 全部评测指标与效率统计的实现 |
| `tool/split_busi_dataset.py` | BUSI 数据集划分 |
| `tool/resize&split/kvasir_split_dataset.py` | Kvasir 数据集划分（默认 8:1:1，可设 seed） |
| `tool/resize&split/add_noise_to_dataset.py` | 为数据集添加指定标准差的高斯噪声（生成噪声鲁棒性测试集） |
| `tool/resize&split/refuge_process_cup.py` | REFUGE 数据集 CUP 掩码处理 |
| `tool/resize&split/refuge_process_geometric_center.py` | REFUGE 数据集按视盘几何中心裁剪 |
| `tool/resize&split/chenpengjie.py` | DICOM 序列读取与窗宽窗位处理 |

## 未包含在本仓库中的内容

出于体积与保密考虑，以下内容未上传：

- `parameterfault/` —— 预训练权重（单文件约 357 MB）
- `work_dir/` —— 训练输出与检查点
- `test_result/` —— 推理结果
- `demo/` —— 演示脚本

## 致谢

本仓库的图像编码器、提示编码器、掩码解码器、TwoWayTransformer 及 `utils/` 下的工具代码改编自 Meta AI 的 [Segment Anything](https://github.com/facebookresearch/segment-anything)，遵循其 Apache-2.0 许可。