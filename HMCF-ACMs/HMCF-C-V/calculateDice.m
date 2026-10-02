function dice = calculateDice(pred_mask, gt_mask)
%==========================================================================
% calculateDice —— 计算 Dice 相似系数（Dice Similarity Coefficient, DSC）
%--------------------------------------------------------------------------
% 功能：
%   定量评价预测分割结果与真实标签（Ground Truth）之间的重合程度，
%   取值范围 [0,1]，越接近 1 表示分割精度越高。
%
% 数学表达式：
%           Dice = 2·|A ∩ B| / (|A| + |B|)
%   其中 A、B 分别为预测前景与真实前景的像素集合，|·| 表示集合基数。
%   等价形式为 Dice = 2·TP / (2·TP + FP + FN)。
%
% 输入：
%   pred_mask : 预测分割二值掩膜（数值/逻辑类型均可）
%   gt_mask   : 真实标签二值掩膜
%
% 输出：
%   dice      : Dice 相似系数；当预测与真实均为空集时约定取 1
%
%==========================================================================

% 统一转换为逻辑类型，保证按像素计数
pred_mask = logical(pred_mask);
gt_mask = logical(gt_mask);

% 交集像素数 |A ∩ B|
intersection = sum(pred_mask(:) & gt_mask(:));

% 预测前景与真实前景的像素总数 |A|、|B|
pred_pixels = sum(pred_mask(:));
gt_pixels = sum(gt_mask(:));

% Dice = 2|A∩B| / (|A| + |B|)
dice = 2 * intersection / (pred_pixels + gt_pixels);

% 边界情形：预测与真实均为全空时约定 Dice = 1
if (pred_pixels + gt_pixels) == 0
    dice = 1;
end
end