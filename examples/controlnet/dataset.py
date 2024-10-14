from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageOps import exif_transpose
import torch
from PIL import Image, ImageDraw
import os
from torchvision.transforms import functional as TF
import torch
import numpy as np
import random
from transformers import AutoTokenizer
import xml.etree.ElementTree as ET
from pathlib import Path
import json
random.seed(10081)

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

class RandomResizedCropWithIoU:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), iou_threshold=0.3, max_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.iou_threshold = iou_threshold
        self.max_attempts = max_attempts
        
        self.crop_top_left = (0, 0)

    def __call__(self, image, mask):
        # for attempt in range(self.max_attempts):
        #     i, j, h, w = self._get_random_params(image)
        #     # cropped_mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)
        #     cropped_mask = TF.crop(mask, i, j, h, w)

        #     if self._compute_iou(cropped_mask, mask) >= self.iou_threshold:
        #         # 如果找到符合要求的裁剪框，裁剪图像和掩码
        #         image = TF.resized_crop(image, i, j, h, w, self.size, Image.BILINEAR)
        #         mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)
        #         self.crop_top_left = (i, j)
        #         return image, mask
        
        # 如果经过多次尝试未能找到合适的裁剪框，返回最后一次的裁剪结果
        image = TF.resize(image,  self.size, Image.BILINEAR)
        mask = TF.resize(mask, self.size, Image.NEAREST)
        self.crop_top_left = (0,0)
        return image, mask

    def _get_random_params(self, image):
        return transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)

    def _compute_iou(self, cropped_mask, original_mask):
        # 计算裁剪后掩码中物体的IoU
        mask_np = np.array(cropped_mask)
        intersection = np.sum(mask_np > 0)
        original_mask_np = np.array(original_mask)
        union = np.sum(original_mask_np > 0)  # 裁剪框的面积（总像素数）
        if union == 0:  # 避免除以零
            return 0
        return intersection / union


class JointTransform(object):
    def __init__(self, 
                 size, 
                 horizontal_flip_prob=0.5,
                 vertical_flip_prob=0.5,
                 scale=(0.5, 1.0), ratio=(3.0/4.0, 4.0/3.0), iou_threshold=0.6, max_attempts=50):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.iou_thrs = iou_threshold
        self.max_attempts = max_attempts
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

        self.transform = RandomResizedCropWithIoU(self.size, self.scale, self.ratio, self.iou_thrs, self.max_attempts)
                
    def __call__(self, image, mask):

        # 随机水平翻转
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 随机垂直翻转
        if random.random() < self.vertical_flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # resize and crop
        image, mask = self.transform(image, mask)
        
        image = self.color_jitter(image)
        
        self.crop_top_left = self.transform.crop_top_left
        
        #  将图像normalize到[0, 1]之间，变换维度[N, C, H, W]
        image = TF.to_tensor(image)
        mask  = TF.to_tensor(mask)
        
        #  将图像normalize到[-1, 1]之间
        image = TF.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        #  binarized image
        mask[mask >= 0.5] = 1
        mask[mask < 0.5]  = 0
        
        return image, mask
    

class SD3InpaintingDataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt, size=(512, 512), exclude_tags=[]):
        
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        
        super().__init__()
        self.custom_instance_prompts = False   # 每张图像使用同一个prompt
        
        self.instance_data_root = Path(instance_data_root)
        self.size = size

        self.instance_image_root = os.path.join(self.instance_data_root, "images")
        self.instance_mask_root  = os.path.join(self.instance_data_root, "masks")

        self.instance_masks_name = sorted([
            fname for fname in os.listdir(self.instance_mask_root)
            if fname.lower().endswith(extensions)
            ])

        self.image_dicts = {}
        for fname in os.listdir(self.instance_image_root):
            if fname.lower().endswith(extensions):
                img_fname, _ = os.path.splitext(fname)
                self.image_dicts[img_fname] = fname
                
        self.instance_prompt = instance_prompt                                 
        self._length = len(self.instance_masks_name)
            
        self.exclude_tags = exclude_tags

        self.image_transforms = JointTransform(self.size, )
        
    def __len__(self):
        return self._length
    
    def perform_transform(self, image, mask):
        instance_image = exif_transpose(image)
        instance_mask  = exif_transpose(mask)
        
        if not image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        if not instance_mask.mode == "L":
            instance_mask = instance_mask.convert("L")
        example = {}
        # 进行数据增强
        example["instance_images"], example["instance_masks"] = self.image_transforms(instance_image, instance_mask)
        example["original_size"] = (instance_image.height, instance_image.width)
        example["crop_top_left"] = self.image_transforms.crop_top_left
        example["instance_prompt"] = self.instance_prompt

        return example
    
    def __getitem__(self, index):
        mask_name = self.instance_masks_name[index % self._length]
        fname = mask_name.replace(".png", "")
        image_name = self.image_dicts[fname]
        
        image_path = os.path.join(self.instance_image_root, image_name)
        mask_path  = os.path.join(self.instance_mask_root, mask_name)
        # 读取数据
        instance_image = Image.open(image_path)
        instance_mask  = Image.open(mask_path)
        
        example = self.perform_transform(image=instance_image, mask=instance_mask,)

        return example  


def sd3_collate_fn(examples):

    pixel_values = [example["instance_images"] for example in examples] # 原图
    masked_pixel_values = [ example["instance_images"] * (example["instance_masks"] < 0.5) for example in examples]
    mask_values  = [example["instance_masks"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    masked_pixel_values = torch.stack(masked_pixel_values)
    mask_values = torch.stack(mask_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()       # images
    masked_pixel_values = masked_pixel_values.to(memory_format=torch.contiguous_format).float()       # masked_pixel_values
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()       # masks

    batch = {
        "pixel_values": pixel_values,   
        "masked_pixel_values": masked_pixel_values,
        "mask_values": mask_values,
        "prompts": prompts,
    }

    return batch