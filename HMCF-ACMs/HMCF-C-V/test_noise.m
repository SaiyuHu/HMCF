%==========================================================================
% test_noise.m —— 抗噪鲁棒性对比实验：HMCF 演化格式 vs. 经典 C-V 模型
%--------------------------------------------------------------------------
% 功能：
%   在受噪声污染的合成/真实图像上，分别以本文提出的 HMCF 数值格式与
%   Chan-Vese (C-V) 模型的显式欧拉格式驱动同一水平集函数演化，
%   并用 Dice 相似系数与平均对称表面距离 (ASSD) 定量评价分割精度。
%
% 实验流程：
%   1) 读入灰度图像并构造 Ground Truth（灰度值为 0 的像素视为目标前景）；
%   2) 按 noise_type 指定的类型/强度对图像施加噪声；
%   3) 以同一初始圆形轮廓初始化 phi_rk（HMCF）与 phi_cv（C-V）；
%   4) 两种格式各迭代 11 步，输出 Dice 与 ASSD 并绘图对比。
%
% 数学模型（水平集演化 PDE，以 C-V 能量泛函的梯度流为例）：
%       ∂φ/∂t = δε(φ) [ - (I - C1)^2 + (I - C2)^2 + μ·κ + ν·P(φ) ]
%   其中 δε(·) 为正则化 Dirac 函数，C1、C2 为演化区域内外的最优
%   灰度拟合常数，κ = -div(∇φ/|∇φ|) 为曲率，P(φ) 为距离正则化惩罚项。
%
%==========================================================================

clear; clc; close all;

%--------------------------------------------------------------------------
% 1. 读取输入图像并转换为灰度 double 型
%--------------------------------------------------------------------------
%img=imread('curve.jpg');           
img=imread('shape3.bmp');
%img=imread('euro-night-lights.jpg');
%img=imread('vessel.bmp');          
%img=imread('twoCells.bmp');      
%img=imread('noisyNonUniform.bmp');
if size(img, 3) > 1
    img = rgb2gray(img);            % 彩色图像灰度化
end
Img = double(img);

%--------------------------------------------------------------------------
% 2. 构造 Ground Truth（真实分割标签）
%    规则：原始图像中灰度值为 0 的像素判定为目标前景（标签记为 1）。
%    注意：GT 在加噪之前由未受污染的图像计算，作为评价基准。
%--------------------------------------------------------------------------
GT = zeros(size(Img));
zero_indices = Img == 0;
GT(zero_indices) = 1;

%--------------------------------------------------------------------------
% 3. 噪声模型选择（每次实验仅启用一种噪声类型）
%    noise_type 可选：'gaussian' | 'speckle' | 'salt_pepper' |
%                     'poisson' | 'periodic' | 'gamma' |
%                     'rayleigh' | 'exponential'
%--------------------------------------------------------------------------
noise_type = 'speckle';         % 默认使用椒盐噪声

switch noise_type
    case 'gaussian'
        % 高斯噪声：零均值、方差 0.1 的加性噪声
        Img = double(imnoise(img, 'Gaussian', 0, 0.1));
        fprintf('测试高斯噪声 (方差=0.1)\n');

    case 'speckle'
        % 斑点（乘性）噪声：强度 0.1
        Img = double(imnoise(img, 'speckle', 0.1));
        fprintf('测试斑点噪声 (方差=0.1)\n');

    case 'salt_pepper'
        % 椒盐噪声：噪声密度 0.05
        Img = double(imnoise(img, 'salt & pepper', 0.05));
        fprintf('测试椒盐噪声 (密度=0.05)\n');

    case 'poisson'
        % 泊松（散粒）噪声：其方差随信号强度变化
        Img = double(imnoise(img, 'poisson'));
        fprintf('测试泊松噪声\n');

    case 'periodic'
        % 周期性噪声：叠加二维正弦干扰条纹（幅值 30，频率 0.1）
        [x, y] = meshgrid(1:size(Img, 2), 1:size(Img, 1));
        u = 0.1; v = 0.1;                        % 空间频率
        periodic_noise = 30 * sin(2 * pi * (u * x + v * y));
        Img = Img + periodic_noise;
        fprintf('测试周期性噪声 (幅值=30, 频率=0.1)\n');

    case 'gamma'
        % Gamma 噪声：作为乘性噪声施加（形状参数 k=2，尺度参数 theta=0.5）
        k = 2; theta = 0.5;
        noise = gamrnd(k, theta, size(Img));
        Img = Img .* (1 + noise / mean(noise(:)) * 0.1);   % 归一化后按 10% 强度调制
        fprintf('测试Gamma噪声 (形状=2, 尺度=0.5, 乘性噪声)\n');

    case 'rayleigh'
        % 瑞利噪声：服从瑞利分布的加性噪声（尺度参数 sigma=5）
        sigma = 5;
        noise = raylrnd(sigma, size(Img));
        Img = Img + noise;
        fprintf('测试瑞利噪声 (sigma=5)\n');

    case 'exponential'
        % 指数噪声：服从指数分布的加性噪声（参数 lambda=0.05，强度按 0.1 缩放）
        lambda = 0.05;
        noise = exprnd(1 / lambda, size(Img));
        Img = Img + noise * 0.1;
        fprintf('测试指数噪声 (lambda=0.05, 强度=0.1)\n');

    otherwise
        % 未识别的类型：回退到默认椒盐噪声
        Img = double(imnoise(img, 'salt & pepper', 0.05));
        fprintf('测试默认椒盐噪声 (密度=0.05)\n');
end

% 将受噪图像灰度值截断回有效区间 [0, 255]
Img = max(0, min(255, Img));

%--------------------------------------------------------------------------
% 4. 初始化水平集函数
%    phi 取圆形符号距离函数 (SDF) 的负值：圆内为负、圆外为正，
%    即初始零水平集为圆心 (ic, jc)、半径 r 的圆周。
%--------------------------------------------------------------------------
[nrow, ncol] = size(Img);
% 初始化用于 salt / gauss 噪声图像的圆形轮廓
ic = nrow / 2;
jc = ncol / 2;
r = 30;
phi = -sdf2circle(nrow, ncol, ic, jc, r);
phi_rk = phi;                       % HMCF 格式的初始水平集
phi_cv = phi;                       % C-V 格式的初始水平集

% 初始化用于 twoCells 图像的圆形轮廓（备用）
% ic = nrow/2; jc = ncol/2; r = 20;
% phi = -sdf2circle(nrow,ncol,ic,jc,r); phi_cv = phi;

% 初始化用于周期性噪声图像的圆形轮廓（备用）
% ic = nrow/2; jc = ncol/2; r = min(ic,jc)-30;
% phi = -sdf2circle(nrow,ncol,ic,jc,r); phi_cv = phi;

%--------------------------------------------------------------------------
% 5. 数值参数设置
%    b        : 双曲曲率流权重（用于内部子步时间步长估计）
%    mu       : 长度项权重
%    delta_t  : 演化总时间步长
%    pu       : 距离正则化（惩罚）项权重
%    epsilon  : Heaviside / Dirac 正则化参数
%    numIter  : 主动轮廓迭代次数
%--------------------------------------------------------------------------
b = 35;
mu = 1000;
delta_t = .1;
pu = 10;
epsilon = 1;
numIter = 50;                      % acm 迭代次数
I = double(Img);                    % 参与演化的图像数据（原图 + 噪声）

%==========================================================================
% 6. HMCF-C-V：采用本文 HMCF 数值格式求解 C-V 能量梯度流
%==========================================================================
tic
for k = 0:numIter
    % 单步演化：先构造演化速度场 phi_t，再经四阶龙格-库塔
    % 空间离散（HMCF）推进一个时间步
    phi_rk = EVOLUTION_RK(I, phi_rk, delta_t, epsilon, b, pu);
end
toc

%==========================================================================
% 7. C-V：采用经典显式欧拉格式求解同一 C-V 模型
%==========================================================================
tic
for k = 0:numIter
    % 单步演化：显式欧拉时间离散 + 曲率长度项 + 数据项 + 惩罚项
    phi_cv = EVOLUTION_CV(I, phi_cv, delta_t, epsilon, mu, pu);
end
toc

%==========================================================================
% 8. 计算 Dice 相似系数与 ASSD 评价指标
%    （零水平集内部 phi >= 0 视为前景，与 GT 比较）
%==========================================================================
% HMCF 结果
HMCF = zeros(size(phi_rk));
zero_indices = phi_rk >= 0;
HMCF(zero_indices) = 1;
dice_HMCF = calculateDice(HMCF, GT);
assd_HMCF = calculateASSD(HMCF, GT);

% C-V 结果
cv = zeros(size(phi_cv));
zero_indices = phi_cv >= 0;
cv(zero_indices) = 1;
dice_cv = calculateDice(cv, GT);
assd_cv = calculateASSD(cv, GT);

% 打印定量结果
fprintf('HMCF-C-V结果:\n');
fprintf('Dice系数: %.4f\n', dice_HMCF);
fprintf('ASSD指标: %.4f\n\n', assd_HMCF);

fprintf('C-V结果:\n');
fprintf('Dice系数: %.4f\n', dice_cv);
fprintf('ASSD指标: %.4f\n', assd_cv);

%==========================================================================
% 9. 结果可视化（可选；如不需要可注释）
%    绿色虚线：初始轮廓；红色实线：算法收敛后的轮廓
%==========================================================================
% HMCF 结果图
figure('Position', [200, 200, 600, 500]);
imagesc(uint8(I));
colormap(gray);
hold on;
% 初始轮廓（绿色虚线）
contour(phi, [0 0], 'g', 'LineWidth', 2, 'LineStyle', '-');
% HMCF 最终轮廓（红色实线）
contour(phi_rk, [0 0], 'r', 'LineWidth', 3);
title(sprintf('HMCF-C-V (Dice=%.4f, ASSD=%.4f)', dice_HMCF, assd_HMCF), ...
    'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', [], 'YTick', []);
box on;
% saveas(gcf, 'period_result/hmcf_255.fig');

% C-V 结果图
figure('Position', [300, 200, 600, 500]);
imagesc(uint8(I));
colormap(gray);
hold on;
% 初始轮廓（绿色虚线）
contour(phi, [0 0], 'g', 'LineWidth', 2, 'LineStyle', '-');
% C-V 最终轮廓（红色实线）
contour(phi_cv, [0 0], 'r', 'LineWidth', 3);
title(sprintf('C-V (Dice=%.4f, ASSD=%.4f)', dice_cv, assd_cv), ...
    'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', [], 'YTick', []);
box on;
% saveas(gcf, 'period_result/cv_255.fig');