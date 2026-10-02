import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
from torchvision.transforms.functional import resize, to_pil_image
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, image_folder, target_folder, membership_folder):
        self.origin_size = None
        self.image_scale_size = None
        self.image_folder = image_folder# "e:/data/images"
        self.target_folder = target_folder

        self.image_files = sorted(os.listdir(image_folder))#["cat.jpg", "dog.jpg", "bird.jpg"]
        self.target_files = sorted(os.listdir(target_folder))
        self.pixel_mean = torch.Tensor([106.7138,  70.5478,  50.7495]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([74.6907, 52.2190, 37.5536]).view(-1, 1, 1)

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww]
    def preprocess(self, image) -> torch.Tensor:
        """
        Expects a numpy with shape HxWxC
        """
        # scaling to 1024x1024
        img_size = 1024
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], img_size)

        image_scale = np.array(resize(to_pil_image(image), target_size))

        input_image_torch = torch.as_tensor(np.array(image_scale))
        # hxWxC -->  cxhxw
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        self.image_scale_size = tuple(input_image_torch.shape[-2:])#对于(3,768,1024)[-2:]返回(768, 1024)
        self.origin_size = tuple(image.shape[0:2])#对于(768,800,3)[0:2]返回(768, 800)
        # Normalize colors
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std

        # Pad to Cx1024x1024

        h, w = input_image_torch.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(input_image_torch, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        '''
        output:
        image:tensor(3,1024,1024)
        image_scale_size:tuple(2)
        origin_size:tuple(2)
        gt:ndarray(H,W)
        '''

        image_path = os.path.join(self.image_folder, self.image_files[idx])
        target_path = os.path.join(self.target_folder, self.target_files[idx])
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_torch = self.preprocess(image)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        target = np.where(target > 0, 1, 0)

        output = {
            "image": image_torch,
            "image_scale_size": self.image_scale_size,
            "origin_size": self.origin_size,
            "gt": torch.FloatTensor(target),
            'image_path':file_name,
        }
        return output


