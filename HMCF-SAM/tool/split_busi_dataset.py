import os
import random
import shutil
from pathlib import Path

def split_busi_dataset():
    """
    将BUSI数据集随机划分为训练集、验证集和测试集
    
    划分比例：
    - 训练集：70%
    - 验证集：15%
    - 测试集：15%
    
    数据集结构：
    - Dataset_BUSI_with_GT/
      - normal/     (正常样本)
      - benign/     (良性样本)
      - malignant/  (恶性样本)
    
    每个类别中的文件命名规则：
    - 图像文件: normal (1).png, benign (1).png, malignant (1).png
    - mask文件: normal (1)_mask.png, benign (1)_mask.png, malignant (1)_mask.png
    """
    
    # 设置随机种子，保证可复现
    seed = 42
    random.seed(seed)
    
    # 数据集根目录
    dataset_root = Path("dataset/Dataset_BUSI_with_GT")
    
    # 输出目录
    output_root = Path("dataset/Dataset_BUSI_split")
    
    # 划分比例
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 创建输出目录结构
    train_dir = output_root / "train" / "image"
    val_dir = output_root / "val" / "image"
    test_dir = output_root / "test" / "image"
    
    train_mask_dir = output_root / "train" / "mask"
    val_mask_dir = output_root / "val" / "mask"
    test_mask_dir = output_root / "test" / "mask"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    train_mask_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)
    test_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储所有样本的信息
    # 每个样本包含: (类别, 基础文件名, 图像路径, mask路径)
    all_samples = []
    
    # 遍历三个类别
    categories = ["normal", "benign", "malignant"]
    
    for category in categories:
        category_dir = dataset_root / category
        if not category_dir.exists():
            print(f"警告: 目录 {category_dir} 不存在")
            continue
        
        # 获取该类别下的所有文件
        files = list(category_dir.glob("*.png"))
        
        # 提取样本基础名称（去掉_mask后缀）
        sample_names = set()
        for file in files:
            # 移除_mask后缀
            name = file.stem
            if name.endswith("_mask"):
                name = name[:-5]  # 移除"_mask"
            sample_names.add(name)
        
        # 为每个样本记录图像和mask路径
        for name in sample_names:
            img_path = category_dir / f"{name}.png"
            mask_path = category_dir / f"{name}_mask.png"
            
            # 只保留同时有图像和mask的样本
            if img_path.exists() and mask_path.exists():
                all_samples.append({
                    'category': category,
                    'name': name,
                    'img_path': img_path,
                    'mask_path': mask_path
                })
    
    print(f"总共找到 {len(all_samples)} 个有效样本（包含图像和mask）")
    print(f"  - normal: {sum(1 for s in all_samples if s['category'] == 'normal')}")
    print(f"  - benign: {sum(1 for s in all_samples if s['category'] == 'benign')}")
    print(f"  - malignant: {sum(1 for s in all_samples if s['category'] == 'malignant')}")
    
    # 打乱样本顺序
    random.shuffle(all_samples)
    
    # 计算划分数量
    total = len(all_samples)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count  # 确保总数不变
    
    print(f"\n划分数量:")
    print(f"  训练集: {train_count} ({train_ratio*100:.1f}%)")
    print(f"  验证集: {val_count} ({val_ratio*100:.1f}%)")
    print(f"  测试集: {test_count} ({test_ratio*100:.1f}%)")
    
    # 划分数据
    train_samples = all_samples[:train_count]
    val_samples = all_samples[train_count:train_count + val_count]
    test_samples = all_samples[train_count + val_count:]
    
    # 复制训练集文件
    print("\n复制训练集文件...")
    for sample in train_samples:
        # 复制图像
        dest_img = train_dir / f"{sample['category']}_{sample['name']}.png"
        shutil.copy2(sample['img_path'], dest_img)
        
        # 复制mask
        dest_mask = train_mask_dir / f"{sample['category']}_{sample['name']}_mask.png"
        shutil.copy2(sample['mask_path'], dest_mask)
    
    # 复制验证集文件
    print("复制验证集文件...")
    for sample in val_samples:
        # 复制图像
        dest_img = val_dir / f"{sample['category']}_{sample['name']}.png"
        shutil.copy2(sample['img_path'], dest_img)
        
        # 复制mask
        dest_mask = val_mask_dir / f"{sample['category']}_{sample['name']}_mask.png"
        shutil.copy2(sample['mask_path'], dest_mask)
    
    # 复制测试集文件
    print("复制测试集文件...")
    for sample in test_samples:
        # 复制图像
        dest_img = test_dir / f"{sample['category']}_{sample['name']}.png"
        shutil.copy2(sample['img_path'], dest_img)
        
        # 复制mask
        dest_mask = test_mask_dir / f"{sample['category']}_{sample['name']}_mask.png"
        shutil.copy2(sample['mask_path'], dest_mask)
    
    # 保存划分规则到文件
    split_info_file = output_root / "split_info.txt"
    with open(split_info_file, 'w', encoding='utf-8') as f:
        f.write("BUSI数据集划分规则\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("划分比例:\n")
        f.write(f"  训练集: {train_ratio*100:.1f}%\n")
        f.write(f"  验证集: {val_ratio*100:.1f}%\n")
        f.write(f"  测试集: {test_ratio*100:.1f}%\n\n")
        
        f.write(f"随机种子: {seed}\n\n")
        
        f.write("划分结果:\n")
        f.write(f"  训练集: {train_count} 个样本\n")
        f.write(f"  验证集: {val_count} 个样本\n")
        f.write(f"  测试集: {test_count} 个样本\n")
        f.write(f"  总计: {total} 个样本\n\n")
        
        f.write("训练集样本列表:\n")
        f.write("-" * 40 + "\n")
        for sample in train_samples:
            f.write(f"  {sample['category']}/{sample['name']}.png\n")
        
        f.write("\n验证集样本列表:\n")
        f.write("-" * 40 + "\n")
        for sample in val_samples:
            f.write(f"  {sample['category']}/{sample['name']}.png\n")
        
        f.write("\n测试集样本列表:\n")
        f.write("-" * 40 + "\n")
        for sample in test_samples:
            f.write(f"  {sample['category']}/{sample['name']}.png\n")
    
    # 统计每个子集的类别分布
    print("\n类别分布统计:")
    
    def count_categories(samples):
        counts = {'normal': 0, 'benign': 0, 'malignant': 0}
        for s in samples:
            counts[s['category']] += 1
        return counts
    
    train_counts = count_categories(train_samples)
    val_counts = count_categories(val_samples)
    test_counts = count_categories(test_samples)
    
    print(f"  训练集: {train_counts}")
    print(f"  验证集: {val_counts}")
    print(f"  测试集: {test_counts}")
    
    print(f"\n数据集划分完成！")
    print(f"输出目录: {output_root}")
    print(f"划分规则文件: {split_info_file}")

if __name__ == "__main__":
    split_busi_dataset()
