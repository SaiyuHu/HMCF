import argparse
import json

import cv2
from PIL import Image
import os
from tqdm import tqdm
import random
join = os.path.join
import torch
import torch.nn as nn
import logging
from build_sam_hmcf import samhmcf_model_registry
from torch.utils.data import DataLoader
from tool.calculate_metric import *
from tool.MyDataset import MyDataset
from segment_anything.lora import LoRA_sam



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the SAM model.")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--model_name", type=str, default='SAMhmcf', help='Model name: SAM, SAMhmcf, or SAMtd')
    parser.add_argument("--checkpoint_folder", type=str, default='work_dir/SAMhmcf_runs/Kvasir/2026-04-15_10.58.04_td=0.2', help="parameterfault")
    parser.add_argument("--data_name", type=str, default="Kvasir", help='Dataset name: Kvasir, Kvasir_noisy_std50, Kvasir_noisy_std100, Refuge, etc.')
    parser.add_argument('--test_folder', type=str, default='test', help='Test folder name: test, val, or train')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    return parser.parse_args()


# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def custom_collate_fn(batch):
    return batch 


def read_td_boundary_length_cache(data_name, sigma=1.0):
    """读取TD边界长度75%分位数缓存"""
    cache_file = os.path.join('cache', 'td_statistics', f"{data_name}_td_75th_percentile.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # 检查缓存是否有效
            if (cached_data.get('sigma') == sigma and 
                'td_75th_percentile' in cached_data):
                
                td_75th_percentile = cached_data['td_75th_percentile']
                print(f"从缓存文件读取TD边界长度75%分位数: {td_75th_percentile:.4f}")
                return td_75th_percentile
                
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"缓存文件读取失败: {e}")
    
    # 如果缓存不存在或读取失败，返回默认值
    default_td_length = 323.0251
    print(f"使用默认TD边界长度: {default_td_length:.4f}")
    return default_td_length


def setup_logging(log_file='evaluation.log'):
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def main():
    seed = 0
    set_seed(seed)
    args = parse_args()
    
    # 根据model_name自动设置if_hmcf和use_td_loss参数
    if args.model_name == 'SAMhmcf':
        args.if_hmcf = True
        args.use_td_loss = False
    elif args.model_name == 'SAMtd':
        args.if_hmcf = False
        args.use_td_loss = True
    else:
        args.if_hmcf = False
        args.use_td_loss = False
    
    result_folder = os.path.join('test_result', args.model_name,args.data_name)
    os.makedirs(result_folder, exist_ok=True)

    logger = setup_logging(log_file=join(result_folder, 'evaluation.log'))
    decoder_checkpoint=join(args.checkpoint_folder, 'decoder_best_parameters')
    lora_checkpoint = join(args.checkpoint_folder, 'lora_best_parameters.pth')
    logger.info(f'Using device {args.device}')
    logger.info(f'Model: {args.model_name}, if_hmcf: {args.if_hmcf}, use_td_loss: {args.use_td_loss}')
    # -------------load SAM----------------------

    sam_parameter = 'parameterfault/sam_vit_b_01ec64.pth'

    # 读取TD边界长度75%分位数缓存（仅当使用HMCF时）
    gt_boundary_length = 323.0251  # 默认值
    if args.if_hmcf:
        gt_boundary_length = read_td_boundary_length_cache(args.data_name)
        logger.info(f"使用TD边界长度75%分位数: {gt_boundary_length:.4f}")
    else:
        logger.info(f"使用默认TD边界长度: {gt_boundary_length:.4f}")

    sam = samhmcf_model_registry['vit_b'](checkpoint=sam_parameter, decoder_parameter=decoder_checkpoint, if_hmcf=args.if_hmcf, gt_boundary_length=gt_boundary_length)
    # Create SAM LoRA
    sam_lora = LoRA_sam(sam, 4)
    sam_lora.load_lora_parameters(lora_checkpoint)
    model = sam_lora.sam
    model.to(args.device)
    test_image_path = join('dataset', args.data_name, args.test_folder, 'image')
    test_mask_path = join('dataset', args.data_name, args.test_folder, 'mask')
    test_dataset = MyDataset(test_image_path, test_mask_path, '')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    # --------------evaluation------------------
    logger.info(f"begin evaluation of {args.data_name}_{args.test_folder} using {args.checkpoint_folder}")
    model.eval()
    
    output_folder = join(result_folder)
    os.makedirs(output_folder, exist_ok=True)
    mask_folder = join(output_folder, 'masks')
    os.makedirs(mask_folder, exist_ok=True)
    
    # 计算效率指标（在正式评价前）
    logger.info("计算分割效率指标...")
    efficiency_metrics = calculate_efficiency_metrics(model, test_loader, args.device, num_runs=5)
    
    # 显示效率指标
    logger.info("分割效率指标:")
    logger.info(f"  平均推理时间: {efficiency_metrics['avg_inference_time_ms']:.2f} ms ± {efficiency_metrics['std_inference_time_ms']:.2f} ms")
    logger.info(f"  帧率 (FPS): {efficiency_metrics['fps']:.2f}")
    logger.info(f"  总参数量: {efficiency_metrics['total_params']:,}")
    logger.info(f"  可训练参数量: {efficiency_metrics['trainable_params']:,}")
    logger.info(f"  模型大小: {efficiency_metrics['model_size_mb']:.2f} MB")
    logger.info(f"  估计FLOPs: {efficiency_metrics['estimated_flops']:,}")
    logger.info(f"  估计Gflops: {efficiency_metrics['estimated_gflops']:.2f}")
    
    iou_scores, dice_scores, sensitivity_scores, specificity_scores, accuracy_scores = 0, 0, 0, 0, 0
    precision_scores = 0
    assd_scores = 0
    hd_scores = 0
    hd95_scores = 0
    betti_scores = 0
    pred_compactness_scores = 0
    gt_compactness_scores = 0
    
    # 边界光滑性指标初始化
    pred_curvature_variance_scores = 0
    pred_boundary_roughness_scores = 0
    pred_fractal_dimension_scores = 0
    pred_curvature_range_scores = 0
    pred_smoothness_scores = 0
    
    gt_curvature_variance_scores = 0
    gt_boundary_roughness_scores = 0
    gt_fractal_dimension_scores = 0
    gt_curvature_range_scores = 0
    gt_smoothness_scores = 0
    
    # 记录开始时间
    import time
    evaluation_start_time = time.time()
    
    with torch.no_grad():
        for i, batched_input in enumerate(tqdm(test_loader)):
            gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
            gt2D = gt2D.to(device=args.device, dtype=torch.float32)

            inputs = to_device(batched_input, args.device)
            output = model(inputs)
            if args.if_hmcf:
                mask = torch.stack([y["mask_logits"] for y in output], dim=0)
            else:
                mask = torch.stack([y["mask_ori"] for y in output], dim=0)

            binary_predictions = (mask > 0).float()
            binary_predictions = binary_predictions.squeeze(0)
            gt2D = gt2D.squeeze(0)

            iou = calculate_iou(binary_predictions, gt2D)
            dice = get_dice(binary_predictions, gt2D)
            sensitivity, specificity, accuracy, precision = compute_metrics(gt2D, binary_predictions)
            assd = calculate_assd(binary_predictions, gt2D)
            hd = calculate_hausdorff_distance(binary_predictions, gt2D)
            hd95 = calculate_hausdorff_distance(binary_predictions, gt2D, percentile=95)
            betti = calculate_betti_error(binary_predictions, gt2D)
            pred_compactness = calculate_compactness(binary_predictions)
            gt_compactness = calculate_compactness(gt2D)
            
            # 计算边界光滑性指标
            pred_smoothness = calculate_boundary_smoothness(binary_predictions)
            gt_smoothness = calculate_boundary_smoothness(gt2D)
            
            logger.info(
                f"Picture [{inputs[0]['image_path']}]: iou-{iou:.4f},dice-{dice:.4f},sensitivity-{sensitivity:.4f},specificity-{specificity:.4f}, "
                f"accuracy-{accuracy:.4f},precision-{precision:.4f},assd-{assd:.4f},hd-{hd:.4f},hd95-{hd95:.4f},betti-{betti:.4f}, "
                f"pred_compact-{pred_compactness:.4f},gt_compact-{gt_compactness:.4f}, "
                f"pred_smooth-{pred_smoothness['smoothness_score']:.4f},gt_smooth-{gt_smoothness['smoothness_score']:.4f}")

            iou_scores = iou_scores + iou
            dice_scores = dice_scores + dice
            sensitivity_scores += sensitivity
            specificity_scores += specificity
            accuracy_scores += accuracy
            precision_scores +=precision
            assd_scores += assd
            hd_scores += hd
            hd95_scores += hd95
            betti_scores += betti
            pred_compactness_scores += pred_compactness
            gt_compactness_scores += gt_compactness
            
            # 累加边界光滑性指标
            pred_curvature_variance_scores += pred_smoothness['curvature_variance']
            pred_boundary_roughness_scores += pred_smoothness['boundary_roughness']
            pred_fractal_dimension_scores += pred_smoothness['fractal_dimension']
            pred_curvature_range_scores += pred_smoothness['curvature_range']
            pred_smoothness_scores += pred_smoothness['smoothness_score']
            
            gt_curvature_variance_scores += gt_smoothness['curvature_variance']
            gt_boundary_roughness_scores += gt_smoothness['boundary_roughness']
            gt_fractal_dimension_scores += gt_smoothness['fractal_dimension']
            gt_curvature_range_scores += gt_smoothness['curvature_range']
            gt_smoothness_scores += gt_smoothness['smoothness_score']
    

            # save the mask
            mask = binary_predictions.mul(255).byte().cpu().numpy()
    
            image = Image.fromarray(mask, mode='L')
            file_name = inputs[0]['image_path']+'.png'
            image.save(join(mask_folder, file_name))
            
            # curvature map saving removed
    
    mean_iou = iou_scores / len(test_loader)
    mean_dice = dice_scores / len(test_loader)
    mean_sensitivity, mean_specificity, mean_accuracy = sensitivity_scores / len(test_loader), specificity_scores / len(
        test_loader), accuracy_scores / len(test_loader)
    mean_precision = precision_scores / len(test_loader)
    mean_assd = assd_scores / len(test_loader)
    mean_hd = hd_scores / len(test_loader)
    mean_hd95 = hd95_scores / len(test_loader)
    mean_betti = betti_scores / len(test_loader)
    mean_pred_compactness = pred_compactness_scores / len(test_loader)
    mean_gt_compactness = gt_compactness_scores / len(test_loader)
    
    # 计算边界光滑性指标均值
    mean_pred_curvature_variance = pred_curvature_variance_scores / len(test_loader)
    mean_pred_boundary_roughness = pred_boundary_roughness_scores / len(test_loader)
    mean_pred_fractal_dimension = pred_fractal_dimension_scores / len(test_loader)
    mean_pred_curvature_range = pred_curvature_range_scores / len(test_loader)
    mean_pred_smoothness = pred_smoothness_scores / len(test_loader)
    
    mean_gt_curvature_variance = gt_curvature_variance_scores / len(test_loader)
    mean_gt_boundary_roughness = gt_boundary_roughness_scores / len(test_loader)
    mean_gt_fractal_dimension = gt_fractal_dimension_scores / len(test_loader)
    mean_gt_curvature_range = gt_curvature_range_scores / len(test_loader)
    mean_gt_smoothness = gt_smoothness_scores / len(test_loader)
    
    logger.info("Evaluation finish")
    logger.info(f"mean_iou - {mean_iou:.4f}, mean_dice - {mean_dice:.4f}, mean_sensitivity-{mean_sensitivity:.4f}, "
                f"mean_specificity-{mean_specificity:.4f}, mean_accuracy-{mean_accuracy:.4f}, mean_precision-{mean_precision:.4f}, "
                f"mean_assd-{mean_assd:.4f}, mean_hd-{mean_hd:.4f}, mean_hd95-{mean_hd95:.4f}, mean_betti-{mean_betti:.4f}, "
                f"mean_pred_compact-{mean_pred_compactness:.4f}, mean_gt_compact-{mean_gt_compactness:.4f}")
    
    # 显示综合评价结果（包含效率指标）
    logger.info("="*60)
    logger.info("综合评价结果")
    logger.info("="*60)
    logger.info("分割精度指标:")
    logger.info(f"  IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}")
    logger.info(f"  敏感性: {mean_sensitivity:.4f}, 特异性: {mean_specificity:.4f}, 准确率: {mean_accuracy:.4f}")
    logger.info(f"  精确率: {mean_precision:.4f}, ASSD: {mean_assd:.4f}")
    logger.info(f"  豪斯多夫距离: {mean_hd:.4f}")
    logger.info(f"  HD95: {mean_hd95:.4f}")
    logger.info(f"  Betti误差: {mean_betti:.4f}")
    logger.info(f"  预测紧凑度: {mean_pred_compactness:.4f}, 真实紧凑度: {mean_gt_compactness:.4f}")
    
    logger.info("\n边界光滑性指标:")
    logger.info(f"  预测光滑度评分: {mean_pred_smoothness:.4f}, 真实光滑度评分: {mean_gt_smoothness:.4f}")
    logger.info(f"  预测曲率方差: {mean_pred_curvature_variance:.4f}, 真实曲率方差: {mean_gt_curvature_variance:.4f}")
    logger.info(f"  预测边界粗糙度: {mean_pred_boundary_roughness:.4f}, 真实边界粗糙度: {mean_gt_boundary_roughness:.4f}")
    logger.info(f"  预测分形维度: {mean_pred_fractal_dimension:.4f}, 真实分形维度: {mean_gt_fractal_dimension:.4f}")
    logger.info(f"  预测曲率范围: {mean_pred_curvature_range:.4f}, 真实曲率范围: {mean_gt_curvature_range:.4f}")
    
    logger.info("\n分割效率指标:")
    logger.info(f"  平均推理时间: {efficiency_metrics['avg_inference_time_ms']:.2f} ms")
    logger.info(f"  帧率 (FPS): {efficiency_metrics['fps']:.2f}")
    logger.info(f"  总参数量: {efficiency_metrics['total_params']:,}")
    logger.info(f"  可训练参数量: {efficiency_metrics['trainable_params']:,}")
    logger.info(f"  模型大小: {efficiency_metrics['model_size_mb']:.2f} MB")
    logger.info(f"  估计FLOPs: {efficiency_metrics['estimated_flops']:,}")
    logger.info(f"  估计Gflops: {efficiency_metrics['estimated_gflops']:.2f}")
    
    # 计算效率-精度平衡指标
    efficiency_score = efficiency_metrics['fps'] * mean_dice  # FPS × Dice系数
    logger.info(f"\n效率-精度平衡指标:")
    logger.info(f"  效率得分 (FPS×Dice): {efficiency_score:.2f}")
    
    # 保存综合评价结果到文件
    summary_file = join(output_folder, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("分割模型综合评价结果\n")
        f.write("="*60 + "\n\n")
        
        f.write("分割精度指标:\n")
        f.write(f"  IoU: {mean_iou:.4f}\n")
        f.write(f"  Dice: {mean_dice:.4f}\n")
        f.write(f"  敏感性: {mean_sensitivity:.4f}\n")
        f.write(f"  特异性: {mean_specificity:.4f}\n")
        f.write(f"  准确率: {mean_accuracy:.4f}\n")
        f.write(f"  精确率: {mean_precision:.4f}\n")
        f.write(f"  ASSD: {mean_assd:.4f}\n")
        f.write(f"  豪斯多夫距离: {mean_hd:.4f}\n")
        f.write(f"  HD95: {mean_hd95:.4f}\n")
        f.write(f"  Betti误差: {mean_betti:.4f}\n")
        f.write(f"  预测紧凑度: {mean_pred_compactness:.4f}\n")
        f.write(f"  真实紧凑度: {mean_gt_compactness:.4f}\n\n")
        
        f.write("分割效率指标:\n")
        f.write(f"  平均推理时间: {efficiency_metrics['avg_inference_time_ms']:.2f} ms\n")
        f.write(f"  帧率 (FPS): {efficiency_metrics['fps']:.2f}\n")
        f.write(f"  总参数量: {efficiency_metrics['total_params']:,}\n")
        f.write(f"  可训练参数量: {efficiency_metrics['trainable_params']:,}\n")
        f.write(f"  模型大小: {efficiency_metrics['model_size_mb']:.2f} MB\n")
        f.write(f"  估计FLOPs: {efficiency_metrics['estimated_flops']:,}\n")
        f.write(f"  估计Gflops: {efficiency_metrics['estimated_gflops']:.2f}\n\n")
        
        f.write("效率-精度平衡指标:\n")
        f.write(f"  效率得分 (FPS×Dice): {efficiency_score:.2f}\n")
    
    logger.info(f"综合评价结果已保存到: {summary_file}")

if __name__ == "__main__":
    main()
