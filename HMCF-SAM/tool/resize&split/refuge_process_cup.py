import os
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
DATASET_ROOT = '/home/dell/hmcf-master/dataset/Refuge2'

# Directories to process
datasets = ['train', 'val', 'test']

def process_cup_only_mask(mask):
    """
    处理掩码文件，只保留视杯（黑色区域）
    - 视杯（黑色，值0）保持不变
    - 视盘（灰色，值128）设置为背景（白色，值255）
    - 背景（白色，值255）保持不变
    """
    # 创建新的掩码，初始化为背景（白色）
    cup_only_mask = np.full_like(mask, 255)
    
    # 只保留视杯区域（黑色，值0）
    cup_only_mask[mask == 0] = 0
    
    return cup_only_mask

def process_dataset(dataset):
    """处理单个数据集的掩码文件"""
    print(f"Processing {dataset} dataset masks...")
    
    # 路径
    dataset_path = os.path.join(DATASET_ROOT, dataset)
    mask_dir = os.path.join(dataset_path, 'mask')  # 原始掩码目录
    cup_only_dir = os.path.join(dataset_path, 'mask_cup_only')  # 输出目录
    
    # 创建输出目录
    os.makedirs(cup_only_dir, exist_ok=True)
    
    # 获取所有掩码文件
    mask_extensions = ['.bmp', '.png', '.jpg', '.jpeg']
    mask_files = []
    
    for ext in mask_extensions:
        files = [f for f in os.listdir(mask_dir) if f.endswith(ext)]
        mask_files.extend(files)
    
    print(f"Found {len(mask_files)} mask files")
    
    # 处理掩码文件
    processed_count = 0
    
    for mask_file in tqdm(mask_files):
        try:
            # 读取掩码文件
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Could not read mask {mask_file}")
                continue
            
            # 处理掩码，只保留视杯
            cup_only_mask = process_cup_only_mask(mask)
            
            # 保存处理后的掩码（保持原始尺寸）
            output_path = os.path.join(cup_only_dir, mask_file)
            cv2.imwrite(output_path, cup_only_mask)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
            continue
    
    print(f"Finished processing {dataset} dataset!")
    print(f"Processed: {processed_count} masks")

def main():
    """主函数，处理所有数据集的掩码文件"""
    for dataset in datasets:
        process_dataset(dataset)
    print("All datasets processed successfully!")
    
    # 验证处理结果
    print("\nVerifying processing results...")
    for dataset in datasets:
        dataset_path = os.path.join(DATASET_ROOT, dataset)
        cup_only_dir = os.path.join(dataset_path, 'mask_cup_only')
        
        if os.path.exists(cup_only_dir):
            files = os.listdir(cup_only_dir)
            print(f"{dataset}: {len(files)} cup-only masks created")
            
            # 检查第一个文件的像素值分布
            if files:
                first_file = files[0]
                mask_path = os.path.join(cup_only_dir, first_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                unique_vals = np.unique(mask)
                print(f"  {first_file}: Unique values - {unique_vals}")
        else:
            print(f"{dataset}: Output directory not found")

if __name__ == "__main__":
    main()
