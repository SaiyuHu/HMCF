import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple
from typing import List, Tuple
from torch.nn import functional as F
from .hmcf_block import MeanCurvatureFlow
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder


class SAM_HMCF(nn.Module):
    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            mask_decoder: MaskDecoder,
            prompt_encoder: PromptEncoder,
            if_hmcf=True,
            gt_boundary_length=None,
    ):
        '''
        if_hmcf: whether to add the hmcf module or not
        gt_boundary_length: TD边界长度的目标值，用于HMCF模块
        '''
        super(SAM_HMCF, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.if_hmcf = if_hmcf
        if self.if_hmcf:
            self.hmcf = MeanCurvatureFlow(gt_boundary_length=gt_boundary_length)
        # freeze encoder
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
            interpolate_mode: str = 'bilinear',
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.size befor padding after scaling
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        # 对于nearest插值模式，不使用align_corners参数
        if interpolate_mode == 'nearest':
            masks = F.interpolate(
                masks,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode=interpolate_mode,
            )
            masks = masks[..., : input_size[0], : input_size[1]]
            masks = F.interpolate(masks, original_size, mode=interpolate_mode)
        else:
            masks = F.interpolate(
                masks,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode=interpolate_mode,
                align_corners=False,
            )
            masks = masks[..., : input_size[0], : input_size[1]]
            masks = F.interpolate(masks, original_size, mode=interpolate_mode, align_corners=False)
        return masks

    def save_parameters(self, save_path: str):
        # save the weight of mask_decoder and hmcf
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.mask_decoder.state_dict(), os.path.join(save_path, 'm_dec_weights.pth'))
        if self.if_hmcf:
            torch.save(self.hmcf.state_dict(), os.path.join(save_path, 'hmcf_weights.pth'))


    def load_parameters(self, save_path: str):
        # load_parameter
        self.mask_decoder.load_state_dict(torch.load(os.path.join(save_path, 'm_dec_weights.pth')))
        if self.if_hmcf:
            self.hmcf.load_state_dict(torch.load(os.path.join(save_path, 'hmcf_weights.pth')))
    def forward(
            self,
            batched_input: List[Dict[str, Any]]
    ):
        """
        batched_input(list(dict)):A list over input information
            each a dictionary with the following keys.
            'images':tensor 3x1024x1024 already transformed
            'image_scale_size':tuple(int,int),image size after scaling but before padding
            'origin_size': tuple(int,int),original size of image,
            "gt": ndarray(H,W) ground truth
            'point_coord': star point coordinates,
            'image_path':file_name
        """

        input_image = torch.stack([x["image"] for x in batched_input], dim=0)  # input_image shape（B，3，1024，1024）
        image_embedding = self.image_encoder(input_image)  # (B, 256, 64, 64)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embedding):
            # do not compute gradients for prompt encoder
            # without prompt
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )

            # mask decoder

            low_res_mask = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),  # (1, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (1, 256, 64, 64)
                multimask_output=False,
            )#(1,1,256,256) 
         
            mask = self.postprocess_masks(low_res_mask, input_size=image_record["image_scale_size"],
                                                   original_size=image_record["origin_size"])#(1,1,H,W)
            
            if self.if_hmcf:

                mask_logits = self.hmcf(mask.squeeze())#(H,W),logits
                output = {
                    'mask_logits': mask_logits,#(H,W)
                }
    
            else:
                output = {
                    'mask_ori': mask.squeeze(),
                }
            outputs.append(output)

        return outputs



