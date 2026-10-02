import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm

class CurvatureMapPreprocessor:
    def __init__(self, mask_folder, map_folder, epsilon=0.1, contrast_parameter=1):
        """
        初始化曲率图预处理器
        
        Args:
            mask_folder: 掩码文件夹路径
            map_folder: 输出文件夹路径
            epsilon: Delta函数中的epsilon参数
            contrast_parameter: 对比度参数
        """
        self.mask_folder = mask_folder
        self.map_folder = map_folder
        self.epsilon = epsilon
        self.contrast_parameter = contrast_parameter
        #5*5的高斯核，扩展为4维 (out_channels, in_channels, H, W)
        gaussian_2d = torch.tensor([[1,  4.,  7.,  4., 1.],
                                    [4., 16., 26., 16., 4.],
                                    [7., 26., 41., 26., 7.],
                                    [4., 16., 26., 16., 4.],
                                    [1.,  4.,  7.,  4., 1.]], dtype=torch.float32) / 273.0
        self.gaussian_kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, 5, 5)
        
        # 创建输出文件夹
        os.makedirs(map_folder, exist_ok=True)
    
    def mask_to_signed_distance_function(self, mask):
        """
        将二值掩码转换为内正外负的符号距离函数
        
        Args:
            mask: 二值掩码图像 (0-1范围)
        Returns:
            sdf: 符号距离函数 (内部为正，外部为负)
        """
        # 确保mask是二值图像
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 计算距离变换
        # 内部距离 (前景到背景的距离)
        dist_inner = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # 外部距离 (背景到前景的距离)
        dist_outer = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 5)
        
        # 创建符号距离函数: 内部为正，外部为负
        sdf = dist_inner - dist_outer
        
        # 归一化到合理范围
        sdf = sdf.astype(np.float32)
        
        return sdf
    
    def Delta(self, phi):
        """
        Delta函数实现
        
        Args:
            phi: 水平集函数
        Returns:
            Delta_h: Delta函数结果
        """
        return (self.epsilon / np.pi) / (self.epsilon**2 + phi**2)
    
    def compute_curvature(self, phi):
        """
        计算水平集函数的曲率
        
        Args:
            phi: 水平集函数
        Returns:
            k: 曲率
        """
        dx = 1
        dy = 1
        
        # 计算梯度
        grad_y, grad_x = np.gradient(phi, dx, dy)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 避免除以零
        grad_mag = np.where(grad_mag < np.finfo(float).eps, np.finfo(float).eps, grad_mag)
        
        # 计算法向分量
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        
        # 计算法向导数的梯度
        dnx_dx, _ = np.gradient(nx, dx, dy)
        _, dny_dy = np.gradient(ny, dx, dy)
        
        # 计算曲率
        k = -(dnx_dx + dny_dy)
        return k
    @staticmethod
    def edge_stop_inverse( grad, K):
        """
        边缘停止逆函数
        
        Args:
            grad: 梯度幅值图像
            K: 对比度参数
        Returns:
            g: 边缘停止函数结果
        """
        return 1.0 / (1.0 + grad / (K**2))
    @staticmethod
    def get_preprocess(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww]
    def preprocess_curvature(self, curvature_edge_stop: np.ndarray) -> torch.Tensor:
        """
        Expects a numpy with shape HxWxC
        """
        # scaling to 1024x1024
        img_size = 1024
        target_size = self.get_preprocess(oldh=curvature_edge_stop.shape[0], oldw=curvature_edge_stop.shape[1], long_side_length=img_size)
        # 使用OpenCV进行缩放，保持数值精度
        map_scale = cv2.resize(curvature_edge_stop, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        input_map_torch = torch.as_tensor(map_scale, dtype=torch.float32)
        
        # Pad to 1024x1024，填充值为0以保持[0,1]范围
        h, w = input_map_torch.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(input_map_torch, (0, padw, 0, padh), value=1)
        
        # 下采样到64x64：使用池化取每个patch的最小值
        # 1024/64 = 16，所以使用16x16的池化窗口
        x_downsampled = -F.max_pool2d(-x.unsqueeze(0), kernel_size=16, stride=16)
        
        return x_downsampled.squeeze(0)   
    def produce_map(self, mask_path):
        """
        处理单个曲率图
        
        Args:
            map_path: 曲率图路径
        """
        # 读取掩码图
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 确保曲率图是归一化的图像
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        
        # 转换为内正外负的符号距离函数
        phi_sdf = self.mask_to_signed_distance_function(mask)
        
        # 转换为水平集函数 phi_gt
        phi_gt = mask.astype(np.float32)
        
        # 计算曲率
        curvature = self.compute_curvature(phi_sdf)
        
        # 计算边界上的曲率
        phi_gt_tensor = torch.from_numpy(phi_gt).unsqueeze(0).unsqueeze(0).float()
        boundary_curvature = F.conv2d(phi_gt_tensor, 
                                 weight=self.gaussian_kernel, 
                                 padding=2).squeeze() * F.conv2d(1-phi_gt_tensor, 
                                 weight=self.gaussian_kernel, 
                                 padding=2).squeeze()*curvature

        # 计算 curvature_sigmoid，映射到[0,1]范围
        curvature_sigmoid = torch.sigmoid(10*boundary_curvature)

        # 获取文件名
        filename = os.path.basename(mask_path)
        name, ext = os.path.splitext(filename)
        
        # 保存为16位PNG图像（适合训练）
        png_filename = os.path.join(self.map_folder, f"{name}.png")
        
        # 将曲率图值映射到16位范围(0-65535)
        map_np = curvature_sigmoid.numpy()

        # 转换为16位PNG
        map_16bit = (map_np * 65535).astype(np.uint16)
        
        cv2.imwrite(png_filename, map_16bit)
        
        return curvature_sigmoid
    def process_all_masks(self, overwrite=True):
        """
        处理所有掩码图像，生成曲率图
        
        Args:
            overwrite: 是否覆盖已存在的曲率图文件（默认：是）
        """
        mask_file = os.listdir(self.mask_folder)
        loop = tqdm(enumerate(mask_file), total=len(mask_file))
        for _, filename in loop:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                name = os.path.splitext(os.path.basename(filename))[0]
                pngname = name+'.png'
                
                # 检查是否跳过已存在的文件
                if not overwrite and pngname in os.listdir(self.map_folder):
                    continue
                
                # print(f'process image{filename}')
                mask_path = os.path.join(self.mask_folder, filename)

                self.produce_map(mask_path)
        print('Finish Processing')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成曲率图')
    parser.add_argument('--dataset_name', type=str, default='Kvasir', help='数据集名称')
    parser.add_argument('--no-overwrite', action='store_false', dest='overwrite', 
                        help='不覆盖已存在的曲率图文件（默认覆盖）')
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'], 
                        help='处理的数据集分割')
    
    # 设置默认参数
    parser.set_defaults(overwrite=True)
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    
    # 定义要处理的子目录
    if args.split == 'all':
        splits = ['test', 'train', 'val']
    else:
        splits = [args.split]
    
    print(f"开始处理数据集: {dataset_name}")
    print(f"处理的分割: {splits}")
    print(f"覆盖模式: {'是' if args.overwrite else '否'}")
    
    # 一次性处理数据集
    for split in splits:
        folder = f'dataset/{dataset_name}/{split}'
        mask_folder = os.path.join(folder, 'mask')
        map_folder = os.path.join(folder, 'curvature_map')
        
        # 检查掩码文件夹是否存在
        if not os.path.exists(mask_folder):
            print(f"警告: 掩码文件夹不存在: {mask_folder}")
            continue
            
        print(f"\n处理分割: {split}")
        print(f"掩码文件夹: {mask_folder}")
        print(f"曲率图文件夹: {map_folder}")
        
        Processor = CurvatureMapPreprocessor(mask_folder=mask_folder,
                                             map_folder=map_folder)
        Processor.process_all_masks(overwrite=args.overwrite)



   
