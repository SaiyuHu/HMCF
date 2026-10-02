# HMCF-C-V

将 **HMCF（Hyperbolic Mean Curvature Flow，双曲平均曲率流）数值格式** 应用于 **Chan–Vese (C-V) 水平集模型** 的图像分割实现，并与 C-V 模型的经典显式欧拉格式进行抗噪鲁棒性对比。

本目录与 [`../../HMCF-SAM/`](../../HMCF-SAM/) 为并列的两条技术路线：

| 路线 | 类型 | 框架 |
| --- | --- | --- |
| HMCF-SAM | 深度学习分割 | SAM + LoRA + HMCF 掩码精修 |
| **HMCF-ACMs / HMCF-C-V** | 变分水平集分割 | 双曲平均曲率流 + C-V 能量泛函 |

## 方法概览

HMCF 将水平集演化由一阶抛物型梯度流推广为**二阶双曲型流**：在 C-V 能量泛函梯度流的基础上引入惯性项，使界面演化以有限"波速"传播（系数 `b` 即波速平方），从而加速收敛并增强对噪声的鲁棒性。

- **空间离散**：四阶紧致（9 点）Laplacian 算子 `LAP`，截断误差 $O(h^4)$
- **时间推进**：截断四阶 Runge–Kutta（对时间 Taylor 展开保留至 $O(\Delta t^2)$），内部子步长由扩散 CFL 条件自适应确定
- **对照组**：C-V 模型的经典一阶显式欧拉格式（`EVOLUTION_CV.m`）

## 数学模型

连续演化方程（C-V 能量泛函梯度流）：

$$\frac{\partial \phi}{\partial t} = \delta_\varepsilon(\phi)\left[-(I-C_1)^2 + (I-C_2)^2 + \mu\kappa + \nu P(\phi)\right]$$

其中

- $\delta_\varepsilon(\cdot)$：正则化 Dirac 函数；
- $C_1, C_2$：演化区域内外的最优灰度拟合常数；
- $\kappa = \nabla\cdot(\nabla\phi/|\nabla\phi|)$：零水平集曲率；
- $P(\phi) = 4\Delta\phi + \kappa$：距离正则化惩罚项。

HMCF 格式进一步将上式视为双曲型算子，对 $\phi$ 与其时间导数 $\phi_t$ 同步推进（$b$ 为波速平方）：

$$\frac{\partial^2 \phi}{\partial t^2} = b\,\Delta\phi$$

## 目录结构与调用关系

```
HMCF-C-V/
├── test_noise.m                 主驱动脚本（抗噪对比实验）
├── EVOLUTION_RK.m               HMCF 格式单步演化（C-V 梯度流）
├── pde_RK4_Truncation4.m        HMCF 数值格式（含局部函数 LAP、BoundNeumann_bc）
├── EVOLUTION_CV.m               C-V 显式欧拉格式（对照组）
├── binaryfit.m                  区域最优灰度拟合常数 C1 / C2
├── Heaviside.m                  正则化 Heaviside 函数 Hε(φ)
├── Delta.m                      正则化 Dirac 函数 δε(φ)
├── curvature.m                  曲率 κ = div(∇φ/|∇φ|)
├── forward_gradient.m           向前差分梯度
├── backward_gradient.m          向后差分梯度
├── sdf2circle.m                 圆形符号距离函数（初始轮廓）
├── calculateDice.m              Dice 相似系数
├── calculateASSD.m              平均对称表面距离 (ASSD)
└── *.bmp / *.jpg                示例图像
```

调用路径：

```
test_noise.m
├── sdf2circle.m                        初始圆形轮廓
├── EVOLUTION_RK.m                      HMCF 格式
│   ├── binaryfit.m → Heaviside.m
│   ├── Delta.m
│   ├── curvature.m → forward_gradient.m / backward_gradient.m
│   └── pde_RK4_Truncation4.m           （局部函数 LAP、BoundNeumann_bc）
├── EVOLUTION_CV.m                      C-V 对照格式
├── calculateDice.m
└── calculateASSD.m
```

## 使用方法

在 MATLAB 中打开本目录，直接运行主脚本：

```matlab
test_noise.m
```

脚本将：构造 Ground Truth → 施加指定噪声 → 以同一初始轮廓分别用 HMCF 与 C-V 演化 → 输出 Dice、ASSD 并绘图对比。

默认图像为 `shape3.bmp`；可在脚本第 1 节取消注释切换为 `curve.jpg`、`vessel.bmp`、`twocells.bmp` 等示例。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `noise_type` | `'salt_pepper'` | 噪声类型，见下表 |
| `b` | 35 | HMCF 波速平方（内部子步长估计用） |
| `mu` | 5000 | 长度项权重 |
| `delta_t` | 0.1 | 演化时间步长 |
| `pu` | 10 | 距离正则化惩罚项权重 |
| `epsilon` | 1 | Heaviside / Dirac 正则化参数 |
| `numIter` | 100 | 主动轮廓迭代次数 |

### 支持的噪声类型

`gaussian`（高斯）、`speckle`（斑点）、`salt_pepper`（椒盐）、`poisson`（泊松）、`periodic`（周期）、`gamma`（Gamma）、`rayleigh`（瑞利）、`exponential`（指数）。

### 评价指标

- **Dice 相似系数**：$2|A\cap B|/(|A|+|B|)$，越接近 1 越好；
- **ASSD（平均对称表面距离）**：边界双向平均距离，越小越好。

## 环境依赖

MATLAB（建议 R2019b 及以上，源文件为 UTF-8 编码）。

- 图像处理工具箱（Image Processing Toolbox）：`imnoise`、`strel`、`imerode`；
- `gamma` / `rayleigh` / `exponential` 噪声分支额外需要统计与机器学习工具箱（`gamrnd`、`raylrnd`、`exprnd`）。

## 致谢

`binaryfit.m`、`Heaviside.m`、`curvature.m`、`forward_gradient.m`、`backward_gradient.m`、`sdf2circle.m` 改编自 Chunming Li 的公开 C-V / DRLSE 参考实现，仅用于学术研究。