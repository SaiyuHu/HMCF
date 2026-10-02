import math
import json
from build_sam_hmcf import samhmcf_model_registry
from tool.MyDataset import MyDataset
import datetime
import logging
import argparse
import os
import numpy as np
import random
join = os.path.join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tool.calculate_metric import calculate_dice_batch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from segment_anything.lora import LoRA_sam
from segment_anything.TD import TDLoss, TDBoundaryLength
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Train the SAM model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--gpu", type=str, default="0", help="Specify GPU block number to use")
    parser.add_argument(
        "--sam_checkpoint", type=str, default="parameterfault/sam_vit_b_01ec64.pth"
    )
    parser.add_argument("--model_name", type=str, default="SAMhmcf", help="model (SAM or SAMhmcf or SAMtd)")
    parser.add_argument("--dataname", type=str, default="small")



    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="weight decay (default: 0.01)"
    )
    return parser.parse_args()


def main():
    seed = 0
    set_seed(seed)
    CONFIG = {'ISIC': (1, 30, 1e-4), 'Refuge': (1, 30, 1e-4), 'Kvasir':(1, 30, 1e-4), 'WHU':(1, 30, 1e-4), 'Kvasir_noisy_std50':(1, 30, 1e-4), 'Kvasir_noisy_std100':(1, 30, 1e-4),'BUSI':(1, 30, 1e-4),'small':(1, 10, 1e-4)}
    args = parse_args()
    
    # Set GPU device if specified
    if args.device == 'cuda' and args.gpu:
        # Only set GPU device if explicitly specified and not using CUDA_VISIBLE_DEVICES
        if args.gpu.strip():
            torch.cuda.set_device(int(args.gpu))
            args.device = f'cuda:{args.gpu}'
    
    if args.model_name == 'SAMhmcf':
        args.if_hmcf = True

    elif args.model_name == 'SAMtd':
        args.if_hmcf = False
        args.use_td_loss = True
    else:
        args.if_hmcf = False
        args.use_td_loss = False
    batch_size, epochs, lr = CONFIG[args.dataname]
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S') 
    model_save_path = os.path.join('work_dir', args.model_name + "_runs", args.dataname, time_str)
    
    os.makedirs(model_save_path, exist_ok=True)
    
    logger = setup_logging(join(model_save_path, 'training.log'))
    logger.info(f'{args}')
    logger.info(f'Using device {args.device}')
    logger.info(f'Train Parameter:\n \
                dataset - {args.dataname}\n \
                epochs - {epochs}\n \
                learning rate - {lr}\n \
                batch size - {batch_size}\n \
                random seed - {seed}')

    # ----------- TensorBoard --------------------
    writer = SummaryWriter(log_dir=join(model_save_path, 'logs'))
    # ------------dataset-------------------
    train_image_path = join('dataset', args.dataname, 'train', 'image')
    train_mask_path = join('dataset', args.dataname, 'train', 'mask')
    val_image_path = join('dataset', args.dataname, 'val', 'image')
    val_mask_path = join('dataset', args.dataname, 'val', 'mask')
    train_dataset = MyDataset(train_image_path, train_mask_path, '')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              collate_fn=custom_collate_fn)

    val_dataset = MyDataset(val_image_path, val_mask_path, '')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                            collate_fn=custom_collate_fn)

    # 设置默认的gt_boundary_length
    gt_boundary_length = 323.0251  # 默认值
    
    if args.if_hmcf:
        # 设置缓存文件路径
        cache_dir = os.path.join('cache', 'td_statistics')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{args.dataname}_td_75th_percentile.json")
        
        print("正在计算训练集TD边界长度75%分位数...")
        td_75th = calculate_td_boundary_length_75th_percentile(train_mask_path, cache_file=cache_file)
        if td_75th is not None:
            gt_boundary_length = td_75th
            logger.info(f"使用计算得到的TD边界长度75%分位数: {gt_boundary_length:.4f}")
        else:
            logger.info(f"使用默认TD边界长度: {gt_boundary_length:.4f}")
    
    # -------------load SAM----------------------
    samhmcf = samhmcf_model_registry['vit_b'](checkpoint=args.sam_checkpoint, if_hmcf=args.if_hmcf, gt_boundary_length=gt_boundary_length)
    # Create SAM LoRA
    sam_lora = LoRA_sam(samhmcf, 4)
    model = sam_lora.sam
    model.to(args.device)
    # ---------------total number of trainable parameters-------------
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'total number of trainable parameters {trainable_num}')

    # -------------loss function and optimizer---------------
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    td_criterion = TDLoss(sigma=1.0)
    
    # 确定是否使用TD loss
    use_td_loss = args.if_hmcf or args.use_td_loss
    if use_td_loss:
        logger.info("使用TD loss训练模式")
    else:
        logger.info("使用标准BCE loss训练模式")
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=Lr_lambda)
    # --------------train----------------------------
    logger.info(f'Parameter weight saved path:{model_save_path}')
    logger.info('Begin training---------------------------')
    BEST_SCORE = 0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, td_criterion, args.device, args.if_hmcf, use_td_loss)
        scheduler.step()
        val_loss, val_score = val(model, val_loader, criterion, td_criterion, calculate_dice_batch, args.device, args.if_hmcf, use_td_loss)

        writer.add_scalar('Training Loss', train_loss, epoch + 1)
        writer.add_scalar('Validation Loss', val_loss, epoch + 1)
        writer.add_scalar('Validation Score', val_score, epoch + 1)
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}]: Train Loss - {train_loss:.4f}, Validation Loss - {val_loss:.4f}, "
            f"Val_score-{val_score:.4f}")
        
        #save the latest model
        sam_lora.save_lora_parameters(join(model_save_path, "lora_latest_parameters.pth"))

        model.save_parameters(join(model_save_path, "decoder_latest_parameters"))

        if val_score > BEST_SCORE:
            BEST_SCORE = val_score
            sam_lora.save_lora_parameters(join(model_save_path, "lora_best_parameters.pth"))

            model.save_parameters(join(model_save_path, "decoder_best_parameters"))
            logger.info(f'BEST {epoch + 1} saved!')
    # close TensorBoard
    writer.close()


def to_device(data, device):
    """Move tensors in a dictionary to device."""
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(element, device) for element in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, optimizer, criterion, td_criterion, device, if_hmcf=True, use_td_loss=False):
    model.train()
    total_loss = 0
    logger = logging.getLogger(__name__)
    loop = tqdm(enumerate(train_loader), total=len(train_loader))

    for step, batched_input in loop:
        optimizer.zero_grad(set_to_none=True)
        gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)#(B,1,H,W)
        gt2D = gt2D.to(device=device, dtype=torch.float32)
        inputs = to_device(batched_input, device)
        if device == "cpu":
            output = model(inputs)
            if if_hmcf:
                mask_logits = torch.stack([y["mask_logits"] for y in output], dim=0)
                bceloss = criterion(mask_logits, gt2D)
                tdloss = td_criterion(mask_logits, gt2D)
                loss = bceloss +0.1*tdloss
            else:
                mask_logits = torch.stack([y["mask_ori"] for y in output], dim=0)
                bceloss = criterion(mask_logits, gt2D)
                if use_td_loss:
                    tdloss = td_criterion(mask_logits, gt2D)
                    loss = bceloss + 0.1 * tdloss
                else:
                    loss = bceloss 
            loss.backward()
            optimizer.step()
        else:
            scaler = GradScaler('cuda')
            output = model(inputs)
            if if_hmcf:
                mask_logits = torch.stack([y["mask_logits"] for y in output], dim=0)
                bceloss = criterion(mask_logits, gt2D)
                tdloss = td_criterion(mask_logits, gt2D)
                loss = bceloss +0.1*tdloss
            else:
                mask_logits = torch.stack([y["mask_ori"] for y in output], dim=0)
                bceloss = criterion(mask_logits, gt2D)
                if use_td_loss:
                    tdloss = td_criterion(mask_logits, gt2D)
                    loss = bceloss + 0.1 * tdloss
                else:
                    loss = bceloss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if if_hmcf or use_td_loss:
            loop.set_postfix(bceloss=bceloss.item(), tdloss=tdloss.item())
        else:
            loop.set_postfix(bceloss=bceloss.item())
        total_loss += loss.item()

    return total_loss / len(train_loader)


def val(model, val_loader, criterion, td_criterion, calculatedice, device, if_hmcf, use_td_loss):
    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for batched_input in val_loader:
            gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
            gt2D = gt2D.to(device=device, dtype=torch.float32)
            inputs = to_device(batched_input, device)
            with autocast('cuda'):
                output = model(inputs)

                if if_hmcf:
                    mask_logits = torch.stack([y["mask_logits"] for y in output], dim=0)
                    bceloss = criterion(mask_logits, gt2D)
                    loss = bceloss 
                else:
                    mask_logits = torch.stack([y["mask_ori"] for y in output], dim=0)
                    bceloss = criterion(mask_logits, gt2D)
                    loss = bceloss  
            
            # 简化loss计算
                dice = calculatedice(mask_logits, gt2D)
                total_loss += loss.item()
                total_dice += dice
    return total_loss / len(val_loader), total_dice / len(val_loader)


def setup_logging(log_file='training.log'):
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 清除现有的handlers
    logger.handlers = []

    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def custom_collate_fn(batch):
    return batch  # return batch list


def Lr_lambda(epoch, warm_up_steps=4, period=4):
    if epoch < warm_up_steps:
        return 0.8 ** (warm_up_steps - epoch)
    else:
        if (epoch - warm_up_steps) < period:
            return (1 + math.cos((epoch - warm_up_steps) * math.pi / period)) / 2
        else:
            return Lr_lambda(epoch - warm_up_steps - period, warm_up_steps=0, period=period * 2)


def calculate_td_boundary_length_75th_percentile(mask_dir, sigma=1.0, cache_file=None):
    """计算训练集mask的TD边界长度75%分位数，支持缓存"""
    
    # 如果提供了缓存文件，尝试读取缓存
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # 检查缓存是否有效（目录和参数匹配）
            if (cached_data.get('mask_dir') == mask_dir and 
                cached_data.get('sigma') == sigma and 
                'td_75th_percentile' in cached_data):
                
                print(f"从缓存文件读取TD边界长度75%分位数: {cached_data['td_75th_percentile']:.4f}")
                return cached_data['td_75th_percentile']
                
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"缓存文件读取失败，重新计算: {e}")
    
    # 检查目录是否存在
    if not os.path.exists(mask_dir):
        print(f"错误: 目录不存在: {mask_dir}")
        return None
    
    # 获取所有mask文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not mask_files:
        print("错误: 未找到mask文件")
        return None
    
    print(f"找到 {len(mask_files)} 个mask文件，计算TD边界长度...")
    
    # 初始化TD计算器
    td_calculator = TDBoundaryLength(sigma=sigma)
    
    td_lengths = []
    
    # 处理每个mask文件
    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)
        
        try:
            # 加载mask
            mask = Image.open(mask_path).convert('L')  # 转换为灰度图
            mask_array = np.array(mask)
            # 转换为二值mask（0或255）
            mask_binary = (mask_array > 128).astype(np.uint8) * 255
            
            # 转换为概率图格式 [1, 1, H, W]
            mask_probs = torch.from_numpy(mask_binary.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            
            # 计算TD边界长度
            td_length = td_calculator(mask_probs)
            td_lengths.append(td_length.item())
            
            # 显示进度
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(mask_files)} 个文件")
                
        except Exception as e:
            print(f"处理文件 {mask_file} 时出错: {e}")
            continue
    
    if not td_lengths:
        print("错误: 未能成功处理任何mask文件")
        return None
    
    # 计算75%分位数
    td_lengths = np.array(td_lengths)
    td_75th_percentile = np.percentile(td_lengths, 75)
    
    print(f"TD边界长度统计:")
    print(f"  样本数量: {len(td_lengths)}")
    print(f"  平均TD边界长度: {np.mean(td_lengths):.4f}")
    print(f"  75%分位数: {td_75th_percentile:.4f}")
    print(f"  中位数: {np.median(td_lengths):.4f}")
    
    # 保存到缓存文件
    if cache_file:
        try:
            cache_data = {
                'mask_dir': mask_dir,
                'sigma': sigma,
                'td_75th_percentile': td_75th_percentile,
                'sample_count': len(td_lengths),
                'mean_td_length': float(np.mean(td_lengths)),
                'median_td_length': float(np.median(td_lengths)),
                'calculation_time': datetime.datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"TD边界长度75%分位数已保存到缓存文件: {cache_file}")
            
        except Exception as e:
            print(f"保存缓存文件失败: {e}")
    
    return td_75th_percentile


if __name__ == "__main__":
    main()
