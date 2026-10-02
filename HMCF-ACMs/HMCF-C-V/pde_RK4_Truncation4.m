function phi = pde_RK4_Truncation4(phi, timeStep, phi_t, b)
%==========================================================================
% pde_RK4_Truncation4 —— HMCF 数值格式：截断四阶 Runge-Kutta 时间推进
%                        + 四阶紧致（9 点）空间离散
%--------------------------------------------------------------------------
% 功能：
%   求解形如
%           ∂φ/∂t = c2·Δφ
%   的抛物型（扩散）子问题，将水平集函数 phi 连同其时间导数 phi_t
%   自当前时刻推进 timeStep。时间方向采用截断四阶 Runge-Kutta格式；空间方向
%   采用四阶紧致 9 点
%   Laplacian 算子 LAP（见文件末尾局部函数）。
%
% 数值格式说明：
%   为保证条件稳定，内部子步长由扩散 CFL 条件控制
%           dt = min( η / sqrt(max(c2)), T ),   η = 0.665 ∈ [0.5, 1]
%   其中 T 为外层要求推进的总时间。若 T 不能被子步长整除，则对剩余
%   时间 dt_rest 再执行一次同步长的推进，以保证推进时间精确等于 T。
%
% 输入：
%   phi      : 当前水平集函数矩阵，尺寸 [nrow, ncol]
%   timeStep : 需要推进的总时间 T
%   phi_t    : 水平集的时间导数（演化速度场）∂φ/∂t
%   b        : 扩散系数 c2（在演化方程中相当于波速平方/扩散率）
%
% 输出：
%   phi      : 推进 timeStep 后的水平集函数（已施加诺伊曼边界条件）
%
%==========================================================================

    % ---- 格式参数 ----
    eta = 0.7;            % 时间权重参数，取值范围 [0.5, 1]
    c_2 = b;                % 扩散系数
    T = timeStep;           % 外层总推进时间

    % ---- 预分配辅助场（避免循环中反复分配内存）----
    [rows, cols] = size(phi);
    D_phi = zeros(rows, cols);
    D_phi_t = zeros(rows, cols);
    phi_t_star = zeros(rows, cols);
    phi_star = zeros(rows, cols);

    % ---- 由 CFL 条件确定内部子步长与子步数 ----
    dt = min(1.04 / sqrt(max(c_2, [], 'all')), T);
    maxiter = floor(T / dt);
    dt_rest = T - maxiter * dt;         % 剩余不足一个子步的时间

    % ---- 主迭代：对每个子步执行截断四阶 RK 推进 ----
    for n = 1:maxiter
        % 当前时刻 φ 与 φ_t 的扩散（Laplacian）响应
        D_phi = c_2 .* LAP(phi);
        D_phi_t = c_2 .* LAP(phi_t);

        % 预测步：由 Taylor 展开构造半步预测值 φ_t^* 与 φ^*
        phi_t_star = phi_t + 0.5 * dt * D_phi + 0.25 * dt^2 * D_phi_t;
        phi_star = phi + 0.5 * dt * (eta * phi_t + (1 - eta) * phi_t_star) ...
                   + 0.25 * dt^2 * D_phi;

        % 预测值处的扩散响应
        D_phi_t_star = c_2 .* LAP(phi_t_star);
        D_phi_star = c_2 .* LAP(phi_star);

        % 校正步：三阶组合（权重 1:2:1）修正 φ_t 与 φ，并补充 O(dt^2) 项
        phi_t = (phi_t + 2 * phi_t_star + dt * (D_phi + D_phi_star)) / 3 ...
                + (dt^2 / 6) * D_phi_t_star;
        phi = (phi + 2 * phi_star + dt * (phi_t + phi_t_star)) / 3 ...
                + (dt^2 / 6) * D_phi_star;
    end

    % ---- 处理剩余时间 dt_rest（若存在），执行最后一次同步长推进 ----
    if dt_rest > 0
        D_phi = c_2 .* LAP(phi);
        D_phi_t = c_2 .* LAP(phi_t);

        phi_t_star = phi_t + 0.5 * dt_rest * D_phi + 0.25 * dt_rest^2 * D_phi_t;
        phi_star = phi + 0.5 * dt_rest * (eta * phi_t + (1 - eta) * phi_t_star) ...
                   + 0.25 * dt_rest^2 * D_phi;

        D_phi_t_star = c_2 .* LAP(phi_t_star);
        D_phi_star = c_2 .* LAP(phi_star);

        phi_t = (phi_t + 2 * phi_t_star + dt_rest * (D_phi + D_phi_star)) / 3 ...
                + (dt_rest^2 / 6) * D_phi_t_star;
        phi = (phi + 2 * phi_star + dt_rest * (phi_t + phi_t_star)) / 3 ...
                + (dt_rest^2 / 6) * D_phi_star;
    end

    % ---- 施加诺伊曼（自由）边界条件 ----
    phi = BoundNeumann_bc(phi);
end

function operator = LAP(f)
%--------------------------------------------------------------------------
% LAP —— 四阶紧致（9 点）Laplacian 空间离散算子
%   采用经典 9 点差分模板逼近 Δf：
%       Δf ≈ [ 4(f_E+f_W+f_N+f_S) + (f_NE+f_NW+f_SE+f_SW) - 20·f_C ] / 6
%   该模板具有 O(h^4) 截断误差（h 为像素网格步长，取 h = 1）。
%--------------------------------------------------------------------------
    [rows, cols] = size(f);
    f_new = zeros(rows, cols);
    i = 2:rows-1;
    j = 2:cols-1;
    f_new(i, j) = ( 4*f(i, j-1) + 4*f(i, j+1) + ...
                        4*f(i-1, j) + 4*f(i+1, j) + ...
                        f(i-1, j-1) + f(i-1, j+1) + ...
                        f(i+1, j-1) + f(i+1, j+1) - ...
                        20*f(i, j) ) / 6;

    operator = BoundNeumann_bc(f_new);
end

function phi = BoundNeumann_bc(phi)
%--------------------------------------------------------------------------
% BoundNeumann_bc —— 诺伊曼（自由）边界条件的虚拟网格点实现
%   令边界点取值等于其内侧相邻点（∂φ/∂n = 0 的一阶离散），
%   以保证边界处零通量。
%--------------------------------------------------------------------------
    % 四条边界：镜像内侧相邻点
    phi(1, :)   = phi(2, :);        % 上边界
    phi(end, :) = phi(end-1, :);    % 下边界
    phi(:, 1)   = phi(:, 2);        % 左边界
    phi(:, end) = phi(:, end-1);    % 右边界

    % 四个角点：取两个相邻边界点的平均值
    phi(1, 1)       = (phi(1, 2) + phi(2, 1)) / 2;
    phi(1, end)     = (phi(1, end-1) + phi(2, end)) / 2;
    phi(end, 1)     = (phi(end-1, 1) + phi(end, 2)) / 2;
    phi(end, end)   = (phi(end-1, end) + phi(end, end-1)) / 2;
end