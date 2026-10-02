import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    随机划分数据集为训练集、验证集和测试集
    
    Args:
        input_dir: 输入数据集目录（包含images和masks子目录）
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子（保证可复现性）
    """
    # 设置随机种子
    random.seed(seed)
    
    # 检查输入目录结构
    images_dir = os.path.join(input_dir, 'images')
    masks_dir = os.path.join(input_dir, 'masks')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"图像目录不存在: {images_dir}")
    if not os.path.exists(masks_dir):
        raise ValueError(f"掩码目录不存在: {masks_dir}")
    
    # 获取图像文件列表
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))])
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 检查对应的掩码文件是否存在
    valid_files = []
    for img_file in image_files:
        mask_file = img_file  # Kvasir_SEG数据集图像和掩码文件名相同
        mask_path = os.path.join(masks_dir, mask_file)
        if os.path.exists(mask_path):
            valid_files.append(img_file)
        else:
            print(f"警告: 未找到 {img_file} 对应的掩码文件")
    
    print(f"有效文件对数量: {len(valid_files)}")
    
    # 随机打乱文件列表
    random.shuffle(valid_files)
    
    # 计算划分点
    total_files = len(valid_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # 划分数据集
    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:val_end]
    test_files = valid_files[val_end:]
    
    print(f"训练集: {len(train_files)} 个文件 ({len(train_files)/total_files*100:.1f}%)")
    print(f"验证集: {len(val_files)} 个文件 ({len(val_files)/total_files*100:.1f}%)")
    print(f"测试集: {len(test_files)} 个文件 ({len(test_files)/total_files*100:.1f}%)")
    
    # 创建输出目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
    
    # 复制文件到相应目录
    def copy_files(files, split_name):
        for file in files:
            # 复制图像
            src_img = os.path.join(images_dir, file)
            dst_img = os.path.join(output_dir, split_name, 'images', file)
            shutil.copy2(src_img, dst_img)
            
            # 复制掩码
            src_mask = os.path.join(masks_dir, file)
            dst_mask = os.path.join(output_dir, split_name, 'masks', file)
            shutil.copy2(src_mask, dst_mask)
    
    # 复制训练集文件
    print("复制训练集文件...")
    copy_files(train_files, 'train')
    
    # 复制验证集文件
    print("复制验证集文件...")
    copy_files(val_files, 'val')
    
    # 复制测试集文件
    print("复制测试集文件...")
    copy_files(test_files, 'test')
    
    # 保存划分信息
    split_info = {
        'total_files': total_files,
        'train_files': len(train_files),
        'val_files': len(val_files),
        'test_files': len(test_files),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_seed': seed,
        'train_files_list': train_files,
        'val_files_list': val_files,
        'test_files_list': test_files
    }
    
    # 保存划分信息到文件
    import json
    split_info_path = os.path.join(output_dir, 'split_info.json')
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"划分信息已保存到: {split_info_path}")
    
    return split_info


def main():
    parser = argparse.ArgumentParser(description='随机划分数据集为训练集、验证集和测试集')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/dell/hmcf-master/dataset/Kvasir_SEG_512x512',
                       help='输入数据集目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/dell/hmcf-master/dataset/Kvasir_SEG_split',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（保证可复现性）')
    
    args = parser.parse_args()
    
    # 检查比例总和是否为1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和应为1.0，当前为: {total_ratio}")
    
    print("开始划分数据集...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分比例: 训练集 {args.train_ratio*100}%, 验证集 {args.val_ratio*100}%, 测试集 {args.test_ratio*100}%")
    print(f"随机种子: {args.seed}")
    
    # 执行划分
    split_info = split_dataset(args.input_dir, args.output_dir, 
                              args.train_ratio, args.val_ratio, args.test_ratio, 
                              args.seed)
    
    print("数据集划分完成!")
    print(f"输出目录结构:")
    print(f"  {args.output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/")
    print(f"  │   └── masks/")
    print(f"  ├── val/")
    print(f"  │   ├── images/")
    print(f"  │   └── masks/")
    print(f"  ├── test/")
    print(f"  │   ├── images/")
    print(f"  │   └── masks/")
    print(f"  └── split_info.json")


if __name__ == "__main__":
    main()