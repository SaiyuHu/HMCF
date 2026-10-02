import os
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
DATASET_ROOT = '/home/dell/hmcf-master/dataset/Refuge2'
CROP_SIZE = 512
HALF_CROP = CROP_SIZE // 2

# Directories to process
datasets = ['train', 'val', 'test']

def calculate_geometric_center(mask):
    """
    基于图像矩计算视杯的几何中心坐标
    使用OpenCV的moments函数计算质心
    """
    # 确保掩码是二值图像（视杯为1，背景为0）
    binary_mask = np.where(mask == 0, 1, 0).astype(np.uint8)
    
    # 计算图像矩
    moments = cv2.moments(binary_mask)
    
    # 计算质心坐标
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        # 如果没有检测到视杯，使用图像中心
        h, w = mask.shape
        center_x = w // 2
        center_y = h // 2
    
    return center_x, center_y

def crop_around_center(image, center_x, center_y, crop_size):
    """
    以给定中心点为中心裁剪图像到指定大小
    处理边界情况，确保裁剪区域不超出图像边界
    """
    h, w = image.shape[:2]
    half_crop = crop_size // 2
    
    # 计算裁剪边界
    x1 = center_x - half_crop
    y1 = center_y - half_crop
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    # 处理边界情况
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h
    
    # 如果裁剪区域小于目标尺寸，进行填充
    if x2 - x1 < crop_size or y2 - y1 < crop_size:
        # 创建目标尺寸的背景
        if len(image.shape) == 3:  # 彩色图像
            cropped = np.zeros((crop_size, crop_size, 3), dtype=image.dtype)
        else:  # 灰度图像
            cropped = np.zeros((crop_size, crop_size), dtype=image.dtype)
        
        # 计算填充偏移
        pad_x1 = max(0, half_crop - center_x)
        pad_y1 = max(0, half_crop - center_y)
        
        # 计算源区域
        src_x1 = max(0, center_x - half_crop)
        src_y1 = max(0, center_y - half_crop)
        src_x2 = min(w, center_x + half_crop)
        src_y2 = min(h, center_y + half_crop)
        
        # 计算目标区域
        dst_x1 = pad_x1
        dst_y1 = pad_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # 复制图像数据
        cropped[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        
        return cropped
    
    return image[y1:y2, x1:x2]

def process_dataset(dataset):
    """处理单个数据集"""
    print(f"Processing {dataset} dataset...")
    
    # 路径
    dataset_path = os.path.join(DATASET_ROOT, dataset)
    image_dir = os.path.join(dataset_path, 'image')
    mask_dir = os.path.join(dataset_path, 'mask')  # 使用mask目录
    
    # 输出目录
    image_out_dir = os.path.join(dataset_path, 'image_geometric_512')
    mask_out_dir = os.path.join(dataset_path, 'mask_geometric_512')
    
    # 创建输出目录
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} image files")
    
    # 根据数据集类型设置mask文件扩展名
    if dataset == 'train':
        mask_extension = '.bmp'
    else:
        mask_extension = '.png'
    
    # 处理文件
    processed_count = 0
    skipped_count = 0
    
    for image_file in tqdm(image_files):
        try:
            # 读取图像
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not read image {image_file}")
                skipped_count += 1
                continue
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 读取对应的掩码文件
            mask_base_name = os.path.splitext(image_file)[0]
            mask_path = os.path.join(mask_dir, mask_base_name + mask_extension)
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {image_file}")
                skipped_count += 1
                continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}")
                skipped_count += 1
                continue
            
            # 计算视杯的几何中心
            center_x, center_y = calculate_geometric_center(mask)
            
            # 以几何中心为中心裁剪图像和掩码
            cropped_image = crop_around_center(image, center_x, center_y, CROP_SIZE)
            cropped_mask = crop_around_center(mask, center_x, center_y, CROP_SIZE)
            
            # 转换图像回BGR格式保存
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            
            # 保存处理后的文件
            cv2.imwrite(os.path.join(image_out_dir, image_file), cropped_image)
            cv2.imwrite(os.path.join(mask_out_dir, image_file.replace('.jpg', '.png')), cropped_mask)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            skipped_count += 1
            continue
    
    print(f"Finished processing {dataset} dataset!")
    print(f"Processed: {processed_count}, Skipped: {skipped_count}")

def main():
    """主函数"""
    for dataset in datasets:
        process_dataset(dataset)
    print("All datasets processed successfully!")
    
    # 验证处理结果
    print("\nVerifying processing results...")
    for dataset in datasets:
        dataset_path = os.path.join(DATASET_ROOT, dataset)
        image_out_dir = os.path.join(dataset_path, 'image_geometric_512')
        mask_out_dir = os.path.join(dataset_path, 'mask_geometric_512')
        
        if os.path.exists(image_out_dir) and os.path.exists(mask_out_dir):
            image_files = os.listdir(image_out_dir)
            mask_files = os.listdir(mask_out_dir)
            
            # 检查图像尺寸
            if image_files:
                first_image = os.path.join(image_out_dir, image_files[0])
                img = cv2.imread(first_image)
                print(f"{dataset}: Images - {len(image_files)} files, Size - {img.shape}")
            
            # 检查掩码尺寸
            if mask_files:
                first_mask = os.path.join(mask_out_dir, mask_files[0])
                mask = cv2.imread(first_mask, cv2.IMREAD_GRAYSCALE)
                print(f"{dataset}: Masks - {len(mask_files)} files, Size - {mask.shape}")
        else:
            print(f"{dataset}: Output directories not found")

if __name__ == "__main__":
    main()
