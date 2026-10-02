function K = curvature(f)
%==========================================================================
% curvature —— 水平集函数的曲率 κ = -div(∇f/|∇f|)
%--------------------------------------------------------------------------
% 功能：
%   计算水平集函数零水平集的曲率，用于长度正则化项与距离正则化惩罚项。
%
% 数学背景：
%   曲率的解析表达式为
%           κ = -div(∇f/|∇f|)
%             = -(fxx·fy^2 + fyy·fx^2 - 2·fx·fy·fxy) / (fx^2 + fy^2)^(3/2)
%   为避免分母在 |∇f|→0 处奇异，数值实现中引入小量 1e-10 进行正则化。
%
% 数值方法：
%   分别以向前差分与向后差分估计梯度 (fx, fy)，构造四个方向组合下的
%   单位法向量并对它们取平均，以增强对噪声的鲁棒性；再对法向量的
%   分量作中心差分 (gradient) 求得散度，从而得到 κ。
%
% 输入：
%   f   : 水平集函数矩阵，尺寸 [nrow, ncol]
%
% 输出：
%   K   : 曲率场，与 f 同尺寸
%
%   Original code: Chunming Li, 04/26/2004.
%==========================================================================

% 向前 / 向后差分估计梯度分量
[f_fx,f_fy] = forward_gradient(f);
[f_bx,f_by] = backward_gradient(f);

% 由四种前/后差分组合构造单位法向量（分母加小量避免奇异）
mag1 = sqrt(f_fx.^2 + f_fy.^2 + 1e-10);
n1x = f_fx./mag1;
n1y = f_fy./mag1;

mag2 = sqrt(f_bx.^2 + f_fy.^2 + 1e-10);
n2x = f_bx./mag2;
n2y = f_fy./mag2;

mag3 = sqrt(f_fx.^2 + f_by.^2 + 1e-10);
n3x = f_fx./mag3;
n3y = f_by./mag3;

mag4 = sqrt(f_bx.^2 + f_by.^2 + 1e-10);
n4x = f_bx./mag4;
n4y = f_by./mag4;

% 对四组法向量求和并归一化（近似中心差分法向量，鲁棒性更好）
nx = n1x + n2x + n3x + n4x;
ny = n1y + n2y + n3y + n4y;

magn = sqrt(nx.^2 + ny.^2);
nx = nx./(magn + 1e-10);
ny = ny./(magn + 1e-10);

% 对法向量分量求偏导并取迹，得到散度 κ = ∂nx/∂x + ∂ny/∂y
[nxx,nxy] = gradient(nx);
[nyx,nyy] = gradient(ny);

K = nxx + nyy;