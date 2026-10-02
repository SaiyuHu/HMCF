import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TDLoss(nn.Module):
    """
    TD损失函数：计算sigmoid(logits)的TDBoundaryLength与gt的相对差异(优化版本)
    
    参数:
        sigma: 高斯核的标准差
        kernel_size: 高斯核大小
    """
    def __init__(self, sigma=1.0, kernel_size=None):
        super(TDLoss, self).__init__()
        self.td_calculator = TDBoundaryLength(sigma=sigma, kernel_size=kernel_size)
        self.eps = 1e-8  # 防止除零的小常数

    def forward(self, pred, target):
        """
        计算TD损失(优化版本)
        
        参数:
            pred: 预测值，可以是logits或概率值，形状为 [1, 1, H, W]
            target: 二值掩膜，形状为 [1, 1, H, W]
        
        返回:
            td_loss: TD损失值
        """
        # 如果输入的pred维度不为[1, 1, H, W]，则转换为[1, 1, H, W]
        if pred.dim() != 4:
            pred = pred.unsqueeze(1)
        
        # 如果输入的target维度不为[1, 1, H, W]，则转换为[1, 1, H, W]
        if target.dim() != 4:
            target = target.unsqueeze(1)
        
        # 计算预测概率的TD边界长度
        pred_probs = torch.sigmoid(pred)
        
        # 计算预测概率的TD边界长度
        pred_td = self.td_calculator(pred_probs)
        
        # 计算目标值的TD边界长度
        target_td = self.td_calculator(target)
    
        # 归一化差异：relu(|pred_td - target_td|) / (pred_td + target_td + eps)
        # 优化：使用更稳定的数值计算
        diff = torch.relu(torch.abs(pred_td - target_td))
        denominator = pred_td + target_td + self.eps
        
        # 使用更稳定的除法
        td_loss = diff / denominator

        return td_loss


class TDBoundaryLength(nn.Module):
    """
    快速TD边界长度计算(进一步优化版本)
    使用近似计算和更高效的实现
    """
    def __init__(self, sigma=1.0, kernel_size=None):
        super(TDBoundaryLength, self).__init__()
        self.sigma = sigma
        self.scale_factor = math.sqrt(math.pi / self.sigma)
        if kernel_size is None:
            kernel_size = min(int(6 * sigma + 1), 15)  # 限制最大核大小
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        self.kernel_size = kernel_size
        self.register_buffer('gaussian_kernel', self.create_gaussian_kernel())
        
    def create_gaussian_kernel(self):
        """创建高斯核"""
        half_size = self.kernel_size // 2
        ax = torch.linspace(-half_size, half_size, self.kernel_size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        gaussian = gaussian / torch.sum(gaussian)
        
        return gaussian.view(1, 1, self.kernel_size, self.kernel_size)
    
    def forward(self, u):
        """边界长度计算"""
        
        # 确保高斯核与输入数据在同一个设备上
        gaussian_kernel = self.gaussian_kernel.to(u.device)
        
        # 使用更高效的实现
        one_minus_u = 1.0 - u
        
        conv_result = F.conv2d(
            one_minus_u, 
            gaussian_kernel, 
            padding=self.kernel_size // 2
        ).squeeze(0).squeeze(0)
        
        # 计算边界长度[batch_size, 1]
        boundary_length = torch.sum(u * conv_result)* self.scale_factor

        return boundary_length