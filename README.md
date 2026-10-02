# HMCF

**HMCF（Hyperbolic Mean Curvature Flow，双曲平均曲率流）** 图像分割代码库。

本仓库包含两条**并列**的技术路线，共享同一核心思想——将传统基于一阶抛物型流的界面/掩码演化推广为二阶双曲型流，通过引入惯性项提升演化速度与抗噪稳定性：

| 路线 | 类型 | 框架 | 目录 |
| --- | --- | --- | --- |
| **HMCF-SAM** | 深度学习分割 | SAM + LoRA 微调 + HMCF 掩码精修 | [`HMCF-SAM/`](HMCF-SAM/) |
| **HMCF-ACMs / HMCF-C-V** | 变分水平集分割 | HMCF 数值格式 + Chan–Vese 能量泛函 | [`HMCF-ACMs/HMCF-C-V/`](HMCF-ACMs/HMCF-C-V/) |

- **HMCF-SAM**：在 Meta 的 Segment Anything (SAM) 上引入 LoRA 轻量微调，并用 HMCF（平均曲率流）模块对 MaskDecoder 输出的掩码做边界精修。
- **HMCF-ACMs / HMCF-C-V**：以四阶紧致空间离散 + 截断四阶 Runge–Kutta 时间推进求解 C-V 能量泛函的双曲型梯度流，并与经典显式欧拉格式作抗噪对比。

## 目录结构

```
HMCF/
├── README.md                       本总览
├── .gitignore
├── HMCF-SAM/                       SAM 深度学习分割（Python / PyTorch）
│   ├── README.md
│   ├── sam_train_with_lora.py      训练入口
│   ├── eval.py                     评测入口
│   ├── build_sam_hmcf.py           模型构建与注册表
│   ├── segment_anything/           SAM 主干及 HMCF 扩展
│   ├── tool/                       数据与指标工具
│   └── cache/                      TD 边界长度统计缓存
└── HMCF-ACMs/
    └── HMCF-C-V/                   变分水平集分割（MATLAB）
        ├── README.md
        ├── test_noise.m            主驱动脚本（抗噪对比实验）
        └── *.m                     演化、数值格式与评价指标
```

## 快速开始

- **HMCF-SAM**（Python / PyTorch）：参见 [`HMCF-SAM/README.md`](HMCF-SAM/README.md)
- **HMCF-C-V**（MATLAB）：参见 [`HMCF-ACMs/HMCF-C-V/README.md`](HMCF-ACMs/HMCF-C-V/README.md)

## 许可

- `HMCF-SAM/segment_anything/` 中的图像编码器、提示编码器、掩码解码器、TwoWayTransformer 及 `utils/` 工具改编自 Meta AI 的 [Segment Anything](https://github.com/facebookresearch/segment-anything)，遵循 Apache-2.0 许可。
- `HMCF-ACMs/HMCF-C-V/` 中的 `binaryfit.m`、`Heaviside.m`、`curvature.m`、`forward_gradient.m`、`backward_gradient.m`、`sdf2circle.m` 改编自 Chunming Li 的公开 C-V / DRLSE 参考实现。
- 本仓库代码仅用于学术研究。