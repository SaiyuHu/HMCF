import torch
import numpy as np
import time
from skimage import segmentation as skimage_seg
import torch.nn.functional as F


def compute_metrics(ground_truth, segmentation_result):
    ground_truth = ground_truth.bool()
    segmentation_result = segmentation_result.bool()

    TP = (ground_truth & segmentation_result).sum().item()
    TN = (~ground_truth & ~segmentation_result).sum().item()
    FP = (~ground_truth & segmentation_result).sum().item()
    FN = (ground_truth & ~segmentation_result).sum().item()

    total_pixels = ground_truth.numel()
    
    # 处理GT全为背景且预测全为背景的特殊情况
    if ground_truth.sum().item() == 0 and segmentation_result.sum().item() == 0:
        # 这种情况下，模型正确预测了全背景
        sensitivity = 1.0  # 没有正例，所以敏感度定义为1
        specificity = 1.0  # 所有负例都正确预测
        accuracy = 1.0     # 所有像素都正确分类
        precision = 1.0    # 没有预测为正例的像素，所以精确度定义为1
    else:
        # 正常情况下的计算
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    return sensitivity, specificity, accuracy, precision



def get_dice(SR, GT):
    """计算Dice系数，正确处理GT全为背景的情况"""
    SR = SR.bool() if not isinstance(SR, torch.BoolTensor) else SR
    GT = GT.bool() if not isinstance(GT, torch.BoolTensor) else GT
    
    intersection = (SR & GT).sum().item()
    sum_SR = SR.sum().item()
    sum_GT = GT.sum().item()
    
    # 处理GT全为背景且预测全为背景的特殊情况
    if sum_GT == 0 and sum_SR == 0:
        return 1.0  # 正确预测全背景，Dice系数定义为1
    
    # 处理分母为0的情况
    if sum_SR + sum_GT == 0:
        return 0.0
    
    dice = 2.0 * intersection / (sum_SR + sum_GT + 1e-10)
    return dice


def calculate_dice_batch(predictions, targets, threshold=0.5, epsilon=1e-8):
    """批量计算Dice系数，符合顶刊标准"""
    predictions = torch.sigmoid(predictions)
    binary_predictions = (predictions > threshold).float()

    batch_dice = []
    for i in range(predictions.size(0)):
        SR = binary_predictions[i].bool()
        GT = targets[i].bool()
        
        intersection = (SR & GT).sum().float()
        sum_SR = SR.sum().float()
        sum_GT = GT.sum().float()
        
        # 处理GT全为背景且预测全为背景的特殊情况
        if sum_GT == 0 and sum_SR == 0:
            dice = torch.tensor(1.0)
        # 处理分母为0的情况
        elif sum_SR + sum_GT == 0:
            dice = torch.tensor(0.0)
        else:
            dice = (2.0 * intersection + epsilon) / (sum_SR + sum_GT + epsilon)
        
        batch_dice.append(dice.item())
    
    average_dice = torch.mean(torch.tensor(batch_dice))
    return average_dice.item()


def calculate_iou(prediction, target):
    """计算IoU，正确处理GT全为背景的情况"""
    prediction = prediction.bool() if not isinstance(prediction, torch.BoolTensor) else prediction
    target = target.bool() if not isinstance(target, torch.BoolTensor) else target
    
    intersection = torch.logical_and(prediction, target).sum().float()
    union = torch.logical_or(prediction, target).sum().float()
    
    # 处理GT全为背景且预测全为背景的特殊情况
    if target.sum().item() == 0 and prediction.sum().item() == 0:
        return 1.0  # 正确预测全背景，IoU定义为1
    
    # 处理分母为0的情况
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou.item()


def calculate_assd(prediction, target):
    """
    计算平均对称表面距离 (Average Symmetric Surface Distance)
    
    参数:
        prediction: 预测的二值分割图，形状为[H, W]
        target: 真实的分割图，形状为[H, W]
        
    返回:
        assd: 平均对称表面距离
    """
    import cv2
    import numpy as np
    
    # 转换为numpy数组
    pred_np = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    # 确保是二值图像
    pred_np = (pred_np > 0.5).astype(np.uint8)
    target_np = (target_np > 0.5).astype(np.uint8)
    
    # 检查是否为空分割
    pred_sum = np.sum(pred_np)
    target_sum = np.sum(target_np)
    
    # 处理GT全为背景且预测全为背景的特殊情况
    if pred_sum == 0 and target_sum == 0:
        # 正确预测了全背景，ASSD定义为0
        return 0.0
    
    # 处理只有一个为空的情况
    if pred_sum == 0 or target_sum == 0:
        # 如果只有一个为空，返回一个较大的距离值（如图像对角线长度）
        # 这符合顶级期刊的处理方式（如Metrics Reloaded框架）
        max_distance = np.sqrt(pred_np.shape[0]**2 + pred_np.shape[1]**2)
        return max_distance
    
    # 计算边界距离变换
    # 预测分割的边界距离
    pred_boundary = cv2.Canny(pred_np, 0, 1)
    pred_dist = cv2.distanceTransform(255 - pred_boundary, cv2.DIST_L2, 5)
    
    # 真实分割的边界距离
    target_boundary = cv2.Canny(target_np, 0, 1)
    target_dist = cv2.distanceTransform(255 - target_boundary, cv2.DIST_L2, 5)
    
    # 计算平均表面距离
    # 预测边界到真实边界的平均距离
    pred_to_target_dist = np.mean(pred_dist[target_boundary > 0]) if np.sum(target_boundary) > 0 else 0
    
    # 真实边界到预测边界的平均距离
    target_to_pred_dist = np.mean(target_dist[pred_boundary > 0]) if np.sum(pred_boundary) > 0 else 0
    
    # 计算对称平均距离
    if np.sum(target_boundary) > 0 and np.sum(pred_boundary) > 0:
        assd = (pred_to_target_dist + target_to_pred_dist) / 2
    elif np.sum(target_boundary) > 0:
        assd = pred_to_target_dist
    elif np.sum(pred_boundary) > 0:
        assd = target_to_pred_dist
    else:
        assd = 0.0
    
    return assd


def calculate_hausdorff_distance(prediction, target, percentile=100):
    """
    计算豪斯多夫距离 (Hausdorff Distance)
    
    参数:
        prediction: 预测的二值分割图，形状为[H, W]
        target: 真实的分割图，形状为[H, W]
        percentile: 百分位数（默认100表示计算完整的豪斯多夫距离，
                   可以设置为95来计算95%豪斯多夫距离）
        
    返回:
        hd: 豪斯多夫距离
    """
    import cv2
    import numpy as np
    
    # 转换为numpy数组
    pred_np = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    # 确保是二值图像
    pred_np = (pred_np > 0.5).astype(np.uint8)
    target_np = (target_np > 0.5).astype(np.uint8)
    
    # 检查是否为空分割
    pred_sum = np.sum(pred_np)
    target_sum = np.sum(target_np)
    
    # 处理GT全为背景且预测全为背景的特殊情况
    if pred_sum == 0 and target_sum == 0:
        # 正确预测了全背景，HD定义为0
        return 0.0
    
    # 处理只有一个为空的情况
    if pred_sum == 0 or target_sum == 0:
        # 如果只有一个为空，返回一个较大的距离值（如图像对角线长度）
        # 这符合顶级期刊的处理方式（如Metrics Reloaded框架）
        max_distance = np.sqrt(pred_np.shape[0]**2 + pred_np.shape[1]**2)
        return max_distance
    
    # 获取边界像素
    pred_boundary = cv2.Canny(pred_np, 0, 1)
    target_boundary = cv2.Canny(target_np, 0, 1)
    
    # 获取边界像素坐标
    pred_coords = np.column_stack(np.where(pred_boundary > 0))
    target_coords = np.column_stack(np.where(target_boundary > 0))
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        max_distance = np.sqrt(pred_np.shape[0]**2 + pred_np.shape[1]**2)
        return max_distance
    
    # 计算从预测边界到真实边界的距离（每个预测边界点到真实边界最近点的距离）
    from scipy.spatial.distance import cdist
    distances_pred_to_target = cdist(pred_coords, target_coords, metric='euclidean')
    min_distances_pred = np.min(distances_pred_to_target, axis=1)
    
    # 计算从真实边界到预测边界的距离
    distances_target_to_pred = cdist(target_coords, pred_coords, metric='euclidean')
    min_distances_target = np.min(distances_target_to_pred, axis=1)
    
    # 计算单向豪斯多夫距离
    hd_pred_to_target = np.percentile(min_distances_pred, percentile)
    hd_target_to_pred = np.percentile(min_distances_target, percentile)
    
    # 返回最大值（豪斯多夫距离）
    hd = max(hd_pred_to_target, hd_target_to_pred)
    
    return hd


def calculate_betti_error(prediction, target):
    """
    计算Betti误差 (Betti Error)
    
    Betti误差衡量分割结果与真实分割在拓扑结构上的差异。
    基于Betti数：
    - Betti-1: 环/洞的数量
    
    参数:
        prediction: 预测的二值分割图，形状为[H, W]
        target: 真实的分割图，形状为[H, W]
        
    返回:
        betti_error: Betti误差（Betti-1的绝对差异）
    """
    import numpy as np
    from skimage import measure
    
    # 转换为numpy数组
    pred_np = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    # 确保是二值图像
    pred_np = (pred_np > 0.5).astype(np.uint8)
    target_np = (target_np > 0.5).astype(np.uint8)
    
    # 如果预测或真实分割全为空，Betti-1=0
    if np.sum(pred_np) == 0 and np.sum(target_np) == 0:
        return 0.0
    
    # 计算预测的Betti-1（环/洞的数量）
    pred_betti_1 = 0
    
    if np.sum(pred_np) > 0:
        contours_pred = measure.find_contours(pred_np, level=0.5)
        pred_betti_1 = len(contours_pred)
    
    # 计算真实的Betti-1（环/洞的数量）
    target_betti_1 = 0
    
    if np.sum(target_np) > 0:
        contours_target = measure.find_contours(target_np, level=0.5)
        target_betti_1 = len(contours_target)
    
    # Betti误差：只使用Betti-1（环/洞数量的绝对差异）
    betti_error = abs(pred_betti_1 - target_betti_1)
    
    return betti_error


def calculate_assd_batch(predictions, targets, threshold=0.5):
    """
    批量计算ASSD
    
    参数:
        predictions: 预测的logits，形状为[B, 1, H, W]或[B, H, W]
        targets: 真实的分割图，形状为[B, 1, H, W]或[B, H, W]
        threshold: 二值化阈值
        
    返回:
        average_assd: 批量平均ASSD
    """
    # 确保是概率值
    if isinstance(predictions, torch.Tensor):
        predictions = torch.sigmoid(predictions)
    
    # 处理不同的维度
    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    if targets.dim() == 4:
        targets = targets.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    
    batch_assd = []
    
    for i in range(predictions.size(0)):
        pred = predictions[i] > threshold
        target = targets[i] > 0.5  # 假设目标已经是二值的
        
        assd_value = calculate_assd(pred, target)
        batch_assd.append(assd_value)
    
    # 计算平均ASSD（包含所有样本，空分割情况已由calculate_assd处理）
    average_assd = np.mean(batch_assd)
    return average_assd


def calculate_compactness(prediction, target=None):
    """
    计算分割结果的紧凑度 (Compactness)
    
    紧凑度 = (周长²) / (4π × 面积)
    
    参数:
        prediction: 预测的分割图，形状为[H, W]
        target: 真实的分割图（可选），如果提供则计算真实分割的紧凑度
        
    返回:
        compactness: 紧凑度值（对于圆形，紧凑度=1；值越大表示形状越不紧凑）
    """
    import cv2
    import numpy as np
    
    # 转换为numpy数组
    pred_np = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    pred_np = (pred_np > 0.5).astype(np.uint8)
    
    # 检查是否为空分割
    pred_sum = np.sum(pred_np)
    
    # 处理GT全为背景且预测全为背景的特殊情况
    if pred_sum == 0:
        # 正确预测了全背景，紧凑度定义为0
        return 0.0
    
    # 计算面积
    area = np.sum(pred_np)
    
    # 计算周长（使用边界轮廓）
    contours, _ = cv2.findContours(pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0.0  # 没有轮廓，返回0
    
    # 计算最大轮廓的周长（处理多个连通区域的情况）
    perimeter = 0
    for contour in contours:
        contour_perimeter = cv2.arcLength(contour, True)
        if contour_perimeter > perimeter:
            perimeter = contour_perimeter
    
    # 计算紧凑度
    if area > 0 and perimeter > 0:
        compactness = (perimeter ** 2) / (4 * np.pi * area)
    else:
        compactness = 0.0
    
    return compactness


def calculate_boundary_smoothness(prediction, target=None):
    """
    计算分割边界的光滑度
    
    参数:
        prediction: 预测的分割图，形状为[H, W]或[1, H, W]
        target: 真实的分割图（可选）
        
    返回:
        smoothness_metrics: 包含各种边界光滑度指标的字典
    """
    import cv2
    import numpy as np
    
    # 转换为numpy数组
    pred_np = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    pred_np = (pred_np > 0.5).astype(np.uint8)
    
    if np.sum(pred_np) == 0:
        return {
            'curvature_variance': 0.0,
            'boundary_roughness': 0.0,
            'fractal_dimension': 0.0,
            'curvature_range': 0.0,
            'smoothness_score': 0.0
        }
    
    # 提取边界轮廓
    contours, _ = cv2.findContours(pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            'curvature_variance': 0.0,
            'boundary_roughness': 0.0,
            'fractal_dimension': 0.0,
            'curvature_range': 0.0,
            'smoothness_score': 0.0
        }
    
    # 找到最大的轮廓
    main_contour = max(contours, key=cv2.contourArea)
    
    # 1. 曲率方差（衡量边界弯曲程度的变化）
    curvature_variance = calculate_curvature_variance(main_contour)
    
    # 2. 边界粗糙度（基于边界点的局部变化）
    boundary_roughness = calculate_boundary_roughness(main_contour)
    
    # 3. 分形维度（衡量边界的复杂程度）
    fractal_dimension = calculate_fractal_dimension(main_contour)
    
    # 4. 曲率范围（最大最小曲率之差）
    curvature_range = calculate_curvature_range(main_contour)
    
    # 5. 综合光滑度评分（值越大越光滑）
    smoothness_score = calculate_smoothness_score(
        curvature_variance, boundary_roughness, fractal_dimension, curvature_range
    )
    
    return {
        'curvature_variance': curvature_variance,
        'boundary_roughness': boundary_roughness,
        'fractal_dimension': fractal_dimension,
        'curvature_range': curvature_range,
        'smoothness_score': smoothness_score
    }


def calculate_curvature_variance(contour):
    """计算边界曲率的方差"""
    if len(contour) < 3:
        return float('inf')
    
    curvatures = []
    for i in range(len(contour)):
        # 计算当前点的曲率
        prev_point = contour[(i-1) % len(contour)][0]
        curr_point = contour[i][0]
        next_point = contour[(i+1) % len(contour)][0]
        
        # 计算向量
        v1 = prev_point - curr_point
        v2 = next_point - curr_point
        
        # 计算夹角（近似曲率）
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            cos_angle = dot_product / (norm_v1 * norm_v2)
            # 限制在有效范围内
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            # 曲率与角度成正比
            curvature = angle
            curvatures.append(curvature)
    
    if len(curvatures) == 0:
        return float('inf')
    
    return np.var(curvatures)


def calculate_boundary_roughness(contour):
    """计算边界粗糙度（基于局部曲率变化）"""
    if len(contour) < 5:
        return float('inf')
    
    roughness = 0
    window_size = min(5, len(contour) // 4)
    
    for i in range(len(contour)):
        # 计算局部曲率变化
        local_curvatures = []
        for j in range(-window_size, window_size + 1):
            idx = (i + j) % len(contour)
            prev_point = contour[(idx-1) % len(contour)][0]
            curr_point = contour[idx][0]
            next_point = contour[(idx+1) % len(contour)][0]
            
            v1 = prev_point - curr_point
            v2 = next_point - curr_point
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 0 and norm_v2 > 0:
                cos_angle = dot_product / (norm_v1 * norm_v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                local_curvatures.append(angle)
        
        if len(local_curvatures) > 1:
            roughness += np.std(local_curvatures)
    
    return roughness / len(contour) if len(contour) > 0 else float('inf')


def calculate_fractal_dimension(contour):
    """计算边界的分形维度（盒计数法）"""
    if len(contour) < 10:
        return float('inf')
    
    # 提取轮廓点坐标
    points = np.array([point[0] for point in contour])
    
    # 归一化坐标
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    range_coords = max_coords - min_coords
    
    if range_coords[0] == 0 or range_coords[1] == 0:
        return float('inf')
    
    points = (points - min_coords) / range_coords
    
    # 盒计数法
    box_sizes = [2, 4, 8, 16, 32]
    box_counts = []
    
    for box_size in box_sizes:
        if box_size > len(points):
            continue
            
        # 创建网格
        grid_size = 1.0 / box_size
        boxes = set()
        
        for point in points:
            box_x = int(point[0] / grid_size)
            box_y = int(point[1] / grid_size)
            boxes.add((box_x, box_y))
        
        box_counts.append(len(boxes))
    
    if len(box_counts) < 2:
        return float('inf')
    
    # 线性回归计算分形维度
    log_sizes = np.log([1.0/s for s in box_sizes[:len(box_counts)]])
    log_counts = np.log(box_counts)
    
    # 防止数值问题
    if np.any(np.isinf(log_sizes)) or np.any(np.isinf(log_counts)):
        return float('inf')
    
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return slope


def calculate_curvature_range(contour):
    """计算边界曲率的范围（最大-最小）"""
    if len(contour) < 3:
        return float('inf')
    
    curvatures = []
    for i in range(len(contour)):
        prev_point = contour[(i-1) % len(contour)][0]
        curr_point = contour[i][0]
        next_point = contour[(i+1) % len(contour)][0]
        
        v1 = prev_point - curr_point
        v2 = next_point - curr_point
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            cos_angle = dot_product / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
    
    if len(curvatures) == 0:
        return float('inf')
    
    return np.max(curvatures) - np.min(curvatures)


def calculate_smoothness_score(curvature_variance, boundary_roughness, fractal_dimension, curvature_range):
    """计算综合光滑度评分"""
    # 归一化各项指标
    # 较小的曲率方差、粗糙度、分形维度和曲率范围表示更光滑
    
    # 防止无穷大值
    if any(np.isinf([curvature_variance, boundary_roughness, fractal_dimension, curvature_range])):
        return 0.0
    
    # 归一化因子（基于经验值）
    norm_curvature_var = min(curvature_variance / 0.1, 1.0)  # 曲率方差越小越好
    norm_roughness = min(boundary_roughness / 0.05, 1.0)     # 粗糙度越小越好
    norm_fractal = min(fractal_dimension / 1.5, 1.0)         # 分形维度越小越好
    norm_range = min(curvature_range / 1.0, 1.0)             # 曲率范围越小越好
    
    # 综合评分（值越大越光滑）
    smoothness_score = 1.0 - (norm_curvature_var + norm_roughness + norm_fractal + norm_range) / 4.0
    
    return max(0.0, smoothness_score)


def calculate_compactness_batch(predictions, targets=None, threshold=0.5):
    """
    批量计算紧凑度
    
    参数:
        predictions: 预测的logits，形状为[B, 1, H, W]或[B, H, W]
        targets: 真实的分割图（可选），形状为[B, 1, H, W]或[B, H, W]
        threshold: 二值化阈值
        
    返回:
        pred_compactness: 预测分割的平均紧凑度
        target_compactness: 真实分割的平均紧凑度（如果targets不为None）
    """
    # 确保是概率值
    if isinstance(predictions, torch.Tensor):
        predictions = torch.sigmoid(predictions)
    
    # 处理不同的维度
    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    
    pred_compactness_list = []
    target_compactness_list = []
    
    for i in range(predictions.size(0)):
        pred = predictions[i] > threshold
        pred_compactness = calculate_compactness(pred)
        pred_compactness_list.append(pred_compactness)
        
        if targets is not None:
            target_i = targets[i] > 0.5
            target_compactness = calculate_compactness(target_i)
            target_compactness_list.append(target_compactness)
    
    # 计算平均值（空分割情况已由calculate_compactness处理）
    avg_pred_compactness = np.mean(pred_compactness_list)
    
    if targets is not None:
        avg_target_compactness = np.mean(target_compactness_list)
        return avg_pred_compactness, avg_target_compactness
    else:
        return avg_pred_compactness


def calculate_efficiency_metrics(model, test_loader, device, num_runs=10):
    """
    计算分割效率指标
    
    参数:
        model: 分割模型
        test_loader: 测试数据加载器
        device: 计算设备（可以是字符串或torch设备对象）
        num_runs: 运行次数（用于计算平均时间）
        
    返回:
        efficiency_metrics: 包含各种效率指标的字典
    """
    
    # 处理设备参数：如果是字符串，转换为torch设备对象
    if isinstance(device, str):
        device = torch.device(device)
    
    # 预热GPU（如果使用GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量推理时间
    model.eval()
    
    # 使用前几个批次进行时间测量
    inference_times = []
    with torch.no_grad():
        for i, batched_input in enumerate(test_loader):
            if i >= num_runs:  # 只测量前num_runs个批次
                break
                
            gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
            gt2D = gt2D.to(device=device, dtype=torch.float32)
            
            # 手动移动输入数据到设备（构建批处理格式）
            inputs = []
            for batch_item in batched_input:
                processed_item = {}
                for key, value in batch_item.items():
                    if isinstance(value, torch.Tensor):
                        processed_item[key] = value.to(device)
                    else:
                        processed_item[key] = value
                inputs.append(processed_item)
            
            # 开始计时
            start_time = time.time()
            
            # 模型推理
            output = model(inputs)
            
            # 结束计时
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
    
    # 计算效率指标
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        fps = 1.0 / avg_inference_time  # 每秒处理的帧数
    else:
        avg_inference_time = std_inference_time = fps = 0.0
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算FLOPs（简化版本，实际需要更复杂的计算）
    # 这里使用一个近似的FLOPs估计
    input_shape = None
    for i, batched_input in enumerate(test_loader):
        if i == 0:
            gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
            input_shape = gt2D.shape
            break
    
    # 简化的FLOPs估计（基于输入尺寸和参数数量）
    if input_shape is not None:
        # 假设每个像素需要进行一定数量的计算
        h, w = input_shape[-2], input_shape[-1]
        estimated_flops = total_params * h * w * 2  # 简化的估计
    else:
        estimated_flops = 0
    
    # 计算Gflops
    estimated_gflops = estimated_flops / (10**9)
    
    efficiency_metrics = {
        'avg_inference_time_ms': avg_inference_time * 1000,  # 转换为毫秒
        'std_inference_time_ms': std_inference_time * 1000,
        'fps': fps,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'estimated_flops': estimated_flops,
        'estimated_gflops': estimated_gflops,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # 参数数量 * 4字节（float32）
    }
    
    return efficiency_metrics


def calculate_memory_usage(model, input_tensor, device):
    """
    计算模型的内存使用情况
    
    参数:
        model: 分割模型
        input_tensor: 输入张量
        device: 计算设备
        
    返回:
        memory_usage: 内存使用情况（MB）
    """
    import torch
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # 记录初始内存使用
        initial_memory = torch.cuda.memory_allocated()
        
        # 进行一次前向传播
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 记录峰值内存使用
        peak_memory = torch.cuda.max_memory_allocated()
        memory_usage = (peak_memory - initial_memory) / (1024 * 1024)  # 转换为MB
        
        torch.cuda.empty_cache()
    else:
        # CPU内存使用估计（简化）
        total_params = sum(p.numel() for p in model.parameters())
        memory_usage = total_params * 4 / (1024 * 1024)  # 参数内存
        
        # 加上激活内存的估计
        if hasattr(input_tensor, 'shape'):
            activation_memory = input_tensor.numel() * 4 / (1024 * 1024)  # 输入张量内存
            memory_usage += activation_memory
    
    return memory_usage










