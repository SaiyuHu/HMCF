import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    与MONAI DiceLoss(sigmoid=True)功能一致的Dice损失函数
    支持logits输入，内部自动应用sigmoid激活
    L = 1 - (2*∑(σ(p)*g) + ε) / (∑σ(p) + ∑g + ε)
    """
    def __init__(self, sigmoid=True, eps=1e-8, reduction='mean'):
        super().__init__()
        self.sigmoid = sigmoid
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        参数:
            pred: 预测值，形状为(N, ...)，可以是logits或概率值
            target: 真实值，形状与pred相同，值在[0,1]区间
        返回:
            损失值
        """
        # 如果sigmoid=True，对预测值应用sigmoid激活
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        
        # 确保目标值在[0,1]范围内
        target = torch.clamp(target, 0.0, 1.0)
        
        # 计算Dice系数
        numerator = 2 * torch.sum(pred * target) + self.eps
        denominator = torch.sum(pred) + torch.sum(target) + self.eps
        
        # 计算损失
        dice_score = numerator / denominator
        loss = 1 - dice_score
        
        return loss