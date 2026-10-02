import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

def add_gaussian_noise_to_image(image, std):
    """
    给图像添加高斯噪声
    
    参数:
        image: PIL Image 或 numpy array，范围 [0, 255]
        std: 噪声标准差 (在 [0, 1] 范围内)
    
    返回:
        noisy_image: 添加噪声后的图像 (PIL Image)
    """
    # 转换为 numpy array 并归一化到 [0, 1]
    if isinstance(image, Image.Image):
        image_np = np.array(image).astype(np.float32) / 255.0
    else:
        image_np = image.astype(np.float32) / 255.0
    
    # 生成高斯噪声
    noise = np.random.normal(0, std, image_np.shape)
    
    # 添加噪声并裁剪到 [0, 1]
    noisy_np = np.clip(image_np + noise, 0, 1)
    
    # 转换回 [0, 255] 范围
    noisy_np = (noisy_np * 255).astype(np.uint8)
    
    # 转换为 PIL Image
    if len(noisy_np.shape) == 2:
        noisy_image = Image.fromarray(noisy_np, mode='L')
    else:
        noisy_image = Image.fromarray(noisy_np, mode='RGB')
    
    return noisy_image

def add_noise_to_dataset(input_dir, output_dir, std, image_folder='image', mask_folder='mask'):
    """
    给数据集添加高斯噪声
    
    参数:
        input_dir: 输入数据集根目录
        output_dir: 输出数据集根目录
        std: 噪声标准差
        image_folder: 图像文件夹名称
        mask_folder: mask文件夹名称
    """
    image_input_path = os.path.join(input_dir, image_folder)
    mask_input_path = os.path.join(input_dir, mask_folder)
    
    image_output_path = os.path.join(output_dir, image_folder)
    mask_output_path = os.path.join(output_dir, mask_folder)
    
    # 创建输出目录
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"正在处理 {len(image_files)} 张图像...")
    
    for filename in tqdm(image_files):
        # 处理图像
        image_path = os.path.join(image_input_path, filename)
        image = Image.open(image_path).convert('RGB')
        noisy_image = add_gaussian_noise_to_image(image, std)
        noisy_image.save(os.path.join(image_output_path, filename))
        
        # 复制mask（不添加噪声）
        mask_path = os.path.join(mask_input_path, filename)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask.save(os.path.join(mask_output_path, filename))
        else:
            # 如果mask不存在，尝试其他扩展名
            base_name = os.path.splitext(filename)[0]
            for ext in ['.jpg', '.png', '.jpeg']:
                mask_path = os.path.join(mask_input_path, base_name + ext)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask.save(os.path.join(mask_output_path, base_name + '.png'))
                    break
    
    print(f"完成！输出目录: {output_dir}")

def main():
    # Kvasir数据集路径
    base='/home/dell/hmcf-master/dataset'
    base_dir = os.path.join(base, 'Kvasir')
    
    # 噪声标准差
    noise_stds = [0.05, 0.1]
    
    # 处理每个噪声水平
    for std in noise_stds:
        # 输出目录
        output_dir = os.path.join(base, f'Kvasir_noisy_std{int(std*1000)}')
        
        print(f"\n{'='*60}")
        print(f"处理噪声标准差: {std}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}")
        
        # 处理训练集
        print("\n处理训练集...")
        add_noise_to_dataset(
            os.path.join(base_dir, 'train'),
            os.path.join(output_dir, 'train'),
            std
        )
        
        # 处理验证集
        print("\n处理验证集...")
        add_noise_to_dataset(
            os.path.join(base_dir, 'val'),
            os.path.join(output_dir, 'val'),
            std
        )
        
        # 处理测试集
        print("\n处理测试集...")
        add_noise_to_dataset(
            os.path.join(base_dir, 'test'),
            os.path.join(output_dir, 'test'),
            std
        )
    
    print(f"\n{'='*60}")
    print("所有噪声数据集生成完成！")
    print(f"{'='*60}")
    print("\n生成的数据集目录:")
    for std in noise_stds:
        output_dir = os.path.join(base_dir, f'Kvasir_noisy_std{int(std*1000)}')
        print(f"  标准差 {std}: {output_dir}")

if __name__ == "__main__":
    main()
