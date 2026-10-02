import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm

class MembershipDegreeProcessor:
    def __init__(self, mask_folder, map_folder):
        """
        初始化曲率图预处理器
        
        Args:
            mask_folder: 掩码文件夹路径
            map_folder: 输出文件夹路径
        """
        self.mask_folder = mask_folder
        self.map_folder = map_folder
        
        # 创建输出文件夹
        os.makedirs(map_folder, exist_ok=True)

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

    def calculate_membership(self, binary_mask, radius):
        """
        计算每个像素点的隶属度 = (邻域内1的个数 - 邻域内0的个数) / 邻域内点的总数
        使用卷积操作加速计算
        
        Args:
            binary_mask: 二值mask图像 (0或1)
            radius: 圆形邻域半径
        Returns:
            membership_map: 隶属度图
        """
        import cv2
        
        H, W = binary_mask.shape
        
        # 创建圆形卷积核
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        # 创建圆形掩码
        y_coords, x_coords = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle_mask = (x_coords**2 + y_coords**2) <= radius**2
        kernel[circle_mask] = 1.0
        
        # 归一化卷积核（每个像素的权重相同）
        kernel = kernel / np.sum(kernel)
        
        # 使用卷积计算1的个数（邻域内1的加权和）
        # 由于binary_mask是0/1值，卷积结果就是邻域内1的个数
        count_ones_map = cv2.filter2D(binary_mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
        
        # 计算邻域内点的总数（边界处会小于完整圆形）
        # 使用全1矩阵与圆形核卷积，得到每个位置的有效点数
        ones_matrix = np.ones_like(binary_mask, dtype=np.float32)
        valid_points_map = cv2.filter2D(ones_matrix, -1, kernel, borderType=cv2.BORDER_REFLECT)
        
        # 避免除零错误
        valid_points_map = np.maximum(valid_points_map, 1e-8)
        
        # 计算0的个数 = 有效点数 - 1的个数
        count_zeros_map = valid_points_map - count_ones_map
        
        # 计算隶属度 = (1的个数 - 0的个数) / 有效点数
        membership_map = (count_ones_map - count_zeros_map) / valid_points_map
        
        return membership_map

    def produce_map(self, mask_path, radius=5):
        """
        处理单个mask图像，计算隶属度图
        
        Args:
            mask_path: mask图像路径
            radius: 圆形邻域半径
        """
        # 读取掩码图
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 确保mask是归一化的图像
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        
        # 二值化mask
        binary_mask = (mask > 0.5).astype(np.float32)
        
        # 计算隶属度图
        membership_map = self.calculate_membership(binary_mask, radius)

        # 映射到[0,1]范围
        membership_normalized = (membership_map + 1) / 2

        # 获取文件名
        filename = os.path.basename(mask_path)
        name, ext = os.path.splitext(filename)
        
        # 保存为16位PNG图像（适合训练）
        png_filename = os.path.join(self.map_folder, f"{name}.png")
        
        # 将隶属度图值映射到16位范围(0-65535)
        map_16bit = (membership_normalized * 65535).astype(np.uint16)
        
        cv2.imwrite(png_filename, map_16bit)
        
        return membership_normalized
    def process_all_masks(self, overwrite=True):
        """
        处理所有掩码图像，生成隶属度图
        
        Args:
            overwrite: 是否覆盖已存在的隶属度图文件（默认：是）
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
    
    parser = argparse.ArgumentParser(description='生成隶属度图')
    parser.add_argument('--dataset_name', type=str, default='Refuge', help='数据集名称')
    parser.add_argument('--no-overwrite', action='store_false', dest='overwrite', 
                        help='不覆盖已存在的隶属度图文件（默认覆盖）')
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
        map_folder = os.path.join(folder, 'membership_map')
        
        # 检查掩码文件夹是否存在
        if not os.path.exists(mask_folder):
            print(f"警告: 掩码文件夹不存在: {mask_folder}")
            continue
            
        print(f"\n处理分割: {split}")
        print(f"掩码文件夹: {mask_folder}")
        print(f"隶属度文件夹: {map_folder}")
        
        Processor = MembershipDegreeProcessor(mask_folder=mask_folder,
                                             map_folder=map_folder)
        Processor.process_all_masks(overwrite=args.overwrite)



   
