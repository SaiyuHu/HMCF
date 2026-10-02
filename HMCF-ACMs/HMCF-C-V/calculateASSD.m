function assd_value = calculateASSD(pred, gt)
%==========================================================================
% calculateASSD —— 计算平均对称表面距离（Average Symmetric Surface
%                  Distance, ASSD）
%--------------------------------------------------------------------------
% 功能：
%   基于分割边界的表面点集，计算预测结果与真实标签之间的平均对称
%   表面距离，用于刻画边界吻合程度；数值越小表示边界越接近。
%
% 数学表达式：
%           ASSD = ( d(A→B) + d(B→A) ) / 2
%           d(A→B) = (1/|A|)·Σ_{a∈A} min_{b∈B} ‖a - b‖₂
%   其中 A、B 分别为预测与真实标签的表面点集，‖·‖₂ 为欧氏距离。
%
% 输入：
%   pred : 预测分割结果（二值图像）
%   gt   : 真实标签（二值图像）
%
% 输出：
%   assd_value : ASSD 值；若任一表面点集为空则返回 Inf
%
%==========================================================================

    % 统一转换为逻辑类型
    pred = logical(pred);
    gt = logical(gt);

    % 分别提取预测与真实标签的表面（边界）点集
    pred_surface = extractSurfacePoints(pred);
    gt_surface   = extractSurfacePoints(gt);

    % 任一表面点集为空时无法计算距离，返回 Inf
    if isempty(pred_surface) || isempty(gt_surface)
        assd_value = Inf;
        return;
    end

    % 双向平均表面距离
    dist_pred_to_gt = calculateSurfaceDistance(pred_surface, gt_surface);   % d(A→B)
    dist_gt_to_pred = calculateSurfaceDistance(gt_surface, pred_surface);   % d(B→A)

    % 对称平均
    assd_value = (dist_pred_to_gt + dist_gt_to_pred) / 2;
end

function surface_points = extractSurfacePoints(binary_mask)
%--------------------------------------------------------------------------
% extractSurfacePoints —— 提取二值掩膜的边界（表面）点坐标
%   通过形态学腐蚀求得内部区域，再用原掩膜减去腐蚀结果得到边界：
%           surface = mask \ erode(mask, se)
%   结构元 se 为 3×3 方形（二维情形）。
%--------------------------------------------------------------------------
    % 3×3 方形结构元
    se = strel('square', 3);

    % 腐蚀得到内部区域
    eroded_mask = imerode(binary_mask, se);

    % 边界点 = 原掩膜 - 腐蚀后的内部
    surface_mask = binary_mask & ~eroded_mask;

    % 返回边界点坐标，格式为 [x, y]（列、行）
    [y, x] = find(surface_mask);
    surface_points = [x, y];
end

function avg_distance = calculateSurfaceDistance(points1, points2)
%--------------------------------------------------------------------------
% calculateSurfaceDistance —— 计算 points1 中每个点到 points2 的最近距离均值
%           d = (1/|P1|)·Σ_{p∈P1} min_{q∈P2} ‖p - q‖₂
%   即单侧平均表面距离。
%--------------------------------------------------------------------------
    if isempty(points1) || isempty(points2)
        avg_distance = Inf;
        return;
    end

    total_distance = 0;

    % 逐点计算到 points2 的最近欧氏距离并累加
    for i = 1:size(points1, 1)
        distances = sqrt(sum((points2 - points1(i, :)).^2, 2));
        min_dist = min(distances);
        total_distance = total_distance + min_dist;
    end

    avg_distance = total_distance / size(points1, 1);
end