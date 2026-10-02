import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ..TD import TDBoundaryLength

class MeanCurvatureFlow(nn.Module):
    """
    平均曲率流模块，支持批处理并行计算
    输入: phi (H, W), curvature (H, W)
    输出: 更新后的phi (H, W)
    """
    
    def __init__(self,
     time_step=0.1, 
     out_iterations=4, 
     in_iterations=3,
     use_checkpoint=True,
     gt_boundary_length=None):
        super(MeanCurvatureFlow, self).__init__()
        self.dT = time_step
        self.out_iterations = out_iterations
        self.in_iterations = in_iterations
        self.dt = self.dT / self.in_iterations

        self.laplacian_kernel = nn.Parameter(torch.tensor([[[[1., 4., 1.],
                                         [4., -20., 4.],
                                         [1., 4., 1.]]]], dtype=torch.float32) / 6.0, requires_grad=False)  
        self.half_dt = 0.5 * self.dt
        self.quarter_dt2 = 0.25 * self.dt**2
        self.one_third = 1.0 / 3.0
        self.one_sixth = 1.0 / 6.0
        self.dt_sq = self.dt**2
        self.eta=.665
        self.use_checkpoint = use_checkpoint

        self.gt_boundary_length = gt_boundary_length
        # 初始化TD边界长度计算器
        self.td_calculator = TDBoundaryLength(sigma=1.0)
        self.eps = 1e-6
    def _mean_curvature_operator(self, phi):
        """
        计算平均曲率算子，使用2D卷积实现
        输入: phi (H, W)
        输出: 平均曲率 (H, W)
        """
        # 添加批次和通道维度: (1, 1, H, W)
        curvature = F.conv2d(phi.unsqueeze(0).unsqueeze(0), weight=self.laplacian_kernel, padding=1)
        return curvature.squeeze(0).squeeze(0)
    def _iteration_step(self, phi, phi_t, b):
        """
        单个迭代步骤的计算，用于梯度检查点
        输入: phi, phi_t, b
        输出: 新的phi, phi_t
        """
        D_phi = b * self._mean_curvature_operator(phi)
        
        phi_t_star = phi_t + self.half_dt * D_phi + self.quarter_dt2 * b * self._mean_curvature_operator(phi_t)
        phi_star = phi + self.half_dt * (self.eta * phi_t + (1 - self.eta) * phi_t_star) + self.quarter_dt2 * D_phi
        
        D_phi_star = b * self._mean_curvature_operator(phi_star)
        
        phi_t_new = (
            (phi_t + 2 * phi_t_star + self.dt * D_phi + self.dt * D_phi_star) * self.one_third +
            self.dt_sq * self.one_sixth * b * self._mean_curvature_operator(phi_t_star)
        )
        phi_new = (
            (phi + 2 * phi_star + self.dt * phi_t + self.dt * phi_t_star) * self.one_third +
            self.dt_sq * self.one_sixth * D_phi_star
        )
        return phi_new, phi_t_new

    def _compute_b_parameter(self, o):
        """
        计算曲率参数b（优化版本）
        
        参数:
            o: 输入特征图，形状为 (H, W)
            
        返回:
            b: 曲率参数，标量值
        """
        # 将输入转换为概率图 [1, 1, H, W]
        prob_map = torch.sigmoid(o).unsqueeze(0).unsqueeze(0)
        
        # 使用TD边界长度计算器计算边界长度
        pre_td = self.td_calculator(prob_map)
    
        # 计算超出比例：max(0, (当前边界长度 - 目标边界长度)) / (当前边界长度 + 目标边界长度 + eps)
        excess_ratio = torch.relu(pre_td - self.gt_boundary_length) / (pre_td + self.gt_boundary_length + self.eps)
        abs_overflow_ratio = torch.max(torch.abs(o))
        # 将超出比例缩放到合适的范围
        b =  abs_overflow_ratio*excess_ratio
    
        return b

    def forward(self, o):
        """
        优化的前向传播版本，使用固定迭代次数
        输入: phi ( H, W), membership_map ( H, W)
        输出: 更新后的phi ( H, W)
        """
        phi = o

        # 曲率参数b的计算
        b = self._compute_b_parameter(o)
        
        # 主迭代循环
        for m in range(self.out_iterations):

            # 计算phi_t
            phi_t = o - torch.tanh(phi)   # (H, W)
            
            for n in range(self.in_iterations):
                if self.use_checkpoint:
                    # 使用梯度检查点包装计算步骤
                    phi, phi_t = checkpoint(
                        self._iteration_step,  # 计算函数
                        phi, phi_t, b,         # 输入参数
                        use_reentrant=False,   # PyTorch 2.0+推荐
                        preserve_rng_state=True
                    )
                else:
                    # 原来的计算
                    phi, phi_t = self._iteration_step(phi, phi_t, b)
        return phi # 返回phi (H, W)

