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
from transformers import CLIPImageProcessor

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
    
    
def sdxl_collate_fn(examples):

    pixel_values = [example["instance_images"] for example in examples] # 原图
    masked_pixel_values = [ example["instance_images"] * (example["instance_masks"] < 0.5) for example in examples]
    mask_values  = [example["instance_masks"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    clip_images = [example["clip_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    clip_images = torch.stack(clip_images)
    masked_pixel_values = torch.stack(masked_pixel_values)
    mask_values = torch.stack(mask_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()       # images
    masked_pixel_values = masked_pixel_values.to(memory_format=torch.contiguous_format).float()       # masked_pixel_values
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()       # masks


    batch = {
        "pixel_values": pixel_values,   
        "clip_images": clip_images,
        "masked_pixel_values": masked_pixel_values,
        "mask_values": mask_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }

    return batch


class BoltSDXLInpaintingIPADataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt, size=(512, 512), exclude_tags=[]):
        
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        super().__init__()
        self.instance_data_root = Path(instance_data_root)
        self.size = size
        # print(f">>> path {self.instance_data_root}")
        # 查找当前文件夹下的所有json文件
        json_files = self.instance_data_root.rglob("./*.json")
        self.json_files = list()
        for file in json_files:
            self.json_files.append(file)    # posixpath
        # print(f">>>>> {self.json_files}")
        image_files = self.instance_data_root.rglob("./*")
        self.image_dicts = {}
        for file_path in image_files:
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                img_fname = file_path.stem
                self.image_dicts[img_fname] = file_path  # posixpath
                
        self.instance_prompt = instance_prompt                                 
        self._length = len(self.json_files)
            
        self.exclude_tags = exclude_tags
        
        self.clip_image_processor = CLIPImageProcessor()

        self.image_transforms = JointTransform(self.size, )
    def __len__(self):
        return self._length
    
    def calculate_iou(self, box1, box2):
        """
        计算两个目标框的IoU。

        参数:
        - box1: tuple, (xmin, ymin, xmax, ymax) 表示的第一个目标框。
        - box2: tuple, (xmin, ymin, xmax, ymax) 表示的第二个目标框。

        返回:
        - iou: float, 两个框的IoU值。
        """
        # 计算交集
        ixmin = max(box1[0], box2[0])
        iymin = max(box1[1], box2[1])
        ixmax = min(box1[2], box2[2])
        iymax = min(box1[3], box2[3])
        
        iw = max(ixmax - ixmin, 0)
        ih = max(iymax - iymin, 0)
        intersection = iw * ih
        
        # 计算并集
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # 计算IoU
        if union == 0:
            return 0
        return intersection / union
    
    def calculate_overlap(self, box1, box2):
        """
        计算两个矩形框的重叠区域坐标。

        参数:
        - box1: tuple, (xmin, ymin, xmax, ymax) 表示的第一个目标框。
        - box2: tuple, (xmin, ymin, xmax, ymax) 表示的第二个目标框（区域）。

        返回:
        - overlap: tuple or None, 重叠区域的坐标 (xmin_overlap, ymin_overlap, xmax_overlap, ymax_overlap)。
                如果没有重叠，返回 None。
        """
        # 计算重叠区域的坐标
        xmin_overlap = max(box1[0], box2[0])
        ymin_overlap = max(box1[1], box2[1])
        xmax_overlap = min(box1[2], box2[2])
        ymax_overlap = min(box1[3], box2[3])
        
        # 检查是否存在重叠区域
        if xmin_overlap < xmax_overlap and ymin_overlap < ymax_overlap:
            # 存在重叠区域，返回坐标
            return (xmin_overlap, ymin_overlap, xmax_overlap, ymax_overlap)
        else:
            # 没有重叠
            return (-1, -1, -1, -1)
        
    def expand_box(self, img_h, img_w, box, expansion_factor=2):
        
        # img_h, img_w = image_shape[:2]
        x_min, y_min, x_max, y_max = box

        # 计算中心点
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # 计算宽和高，并找到最大值
        width = x_max - x_min
        height = y_max - y_min
        max_dim = max(width, height)
        if max_dim > 512:
            return x_min, y_min, x_max, y_max

        # if max_dim < 50:
        #     expansion_factor = 10
        
        # elif max_dim < 70:
        #     expansion_factor = 8
        
        # elif max_dim < 90:
        #     expansion_factor = 6

        # 计算扩展边界
        new_half_dim = int(max_dim * expansion_factor // 2)
        new_x_min = max(center_x - new_half_dim, 0)
        new_y_min = max(center_y - new_half_dim, 0)
        new_x_max = min(center_x + new_half_dim, img_w)
        new_y_max = min(center_y + new_half_dim, img_h)

        return new_x_min, new_y_min, new_x_max, new_y_max
    
    def crop_and_expand_with_masks(self, image, boxes, iou_threshold=0.5, expansion_factor=5):
        """
        参数:
        - image: PIL.Image 对象, 输入图像。
        - boxes: list of tuples, 每个元组为 (xmin, ymin, xmax, ymax) 表示一个目标框。
        - expand_ratio: float, 扩展边框的系数。
        - iou_threshold: float, IoU阈值，控制哪些框生成mask。

        返回:
        - cropped_image: PIL.Image 对象, 裁剪后的图像。
        - masks: list of PIL.Image 对象, 对应扩展区域内目标框的 mask 列表。
        """
        # 随机选择一个目标框
        selected_box = random.choice(boxes)
        # xmin, ymin, xmax, ymax = selected_box
        img_width, img_height = image.size
        new_xmin, new_ymin, new_xmax, new_ymax = self.expand_box(img_height, img_width, selected_box, expansion_factor)
        
        # 从图像中裁剪出扩展后的区域
        cropped_image = image.crop((new_xmin, new_ymin, new_xmax, new_ymax))
        
        # 生成扩展区域内的所有符合IoU条件的目标框的 mask
        mask = Image.new("L", image.size, 0)
        expanded_box = (new_xmin, new_ymin, new_xmax, new_ymax)
        for box in boxes:
            overlap_box = self.calculate_overlap(box, expanded_box)
            
            iou = self.calculate_iou(box, overlap_box)
            if iou >= iou_threshold:
                # 创建黑色背景的mask
                
                draw = ImageDraw.Draw(mask)
                draw.rectangle([box[0], box[1], box[2], box[3]], fill=255)  # 用白色填充目标框区域
                
        # 裁剪 mask 与扩展区域一致
        mask = mask.crop((new_xmin, new_ymin, new_xmax, new_ymax))
        
        return cropped_image, mask
    
    def perform_transform(self, image, mask):
        instance_image = exif_transpose(image)
        instance_mask  = exif_transpose(mask)
        
        if not image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        if not instance_mask.mode == "L":
            instance_mask = instance_mask.convert("L")
        example = {}
        # 进行数据增强
        example["clip_images"] = self.clip_image_processor(images=instance_image, return_tensors="pt").pixel_values[0]
        example["instance_images"], example["instance_masks"] = self.image_transforms(instance_image, instance_mask)
        example["original_size"] = (instance_image.height, instance_image.width)
        example["crop_top_left"] = self.image_transforms.crop_top_left
        example["instance_prompt"] = self.instance_prompt

        return example
    
    def __getitem__(self, index):
        
        json_path = self.json_files[index % self._length]
        img_fname = json_path.stem
        img_path  = str(self.image_dicts[img_fname])
        
        # 解析json文件得到目标框信息
        with open(str(json_path)) as f:
            json_data = json.load(f)
        bboxes = list()
        for item in json_data["shapes"]:
            name = item["label"]
            if name in self.exclude_tags: continue
            points = item["points"]
            xmin, ymin, xmax, ymax = points[0][0], points[0][1], points[1][0], points[1][1]
            bboxes.append([xmin, ymin, xmax, ymax])
            
        # 扩边并得到image和mask
        if len(bboxes) == 0:
            return self.__getitem__(index+1)
        
        # 读取数据
        instance_image = Image.open(img_path)
        instance_image, instance_mask = self.crop_and_expand_with_masks(image=instance_image, boxes=bboxes, expansion_factor=3)
        # instance_image.save(f"datasets/zaohe/tmp/images/{img_fname}.png")
        # instance_mask.save(f"datasets/zaohe/tmp/masks/{img_fname}.png")
        
        example = self.perform_transform(image=instance_image, mask=instance_mask,)

        return example   

if __name__ == "__main__":

    instance_root = "datasets/anomaly_data_0708_0815"
    train_dataset = BoltSDXLInpaintingIPADataset(
        instance_root,
        instance_prompt="a missing sks bolt.",
        # tokenizer=tokenizer,
        # class_data_root=None,
        size=(512, 512),
    )


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda examples: sdxl_collate_fn(examples,),
        num_workers=1,
    )
    
    for step, batch in enumerate(train_dataloader):
        # print(batch["pixel_values"].shape)
        # print(batch["masked_pixel_values"].shape)
        # print(batch["mask_values"].shape)
        # mask_values = batch["mask_values"]
        # masks = torch.nn.functional.interpolate(mask_values, size=(64, 64))
        pass
        # break