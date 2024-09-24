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

class DreamBoothInpaintingDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,                                  # 分词器
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=(512, 512),
        center_crop=False,
        encoder_hidden_states=None,                 # text_embedding
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = (instance_data_root)
        if not os.path.exists(self.instance_data_root):
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_image_root = os.path.join(self.instance_data_root, "images")
        self.instance_mask_root  = os.path.join(self.instance_data_root, "masks")
        
        self.instance_images_name = sorted([
            fname for fname in os.listdir(self.instance_image_root)
            if fname.lower().endswith(extensions)
            ])        # 获取到图像数据
        self.num_instance_images = len(self.instance_images_name)                   # 得到图像数据的个数
        self.instance_masks_name = sorted([
            fname for fname in os.listdir(self.instance_mask_root)
            if fname.lower().endswith(extensions)
            ])
        assert len(self.instance_images_name) == len(self.instance_masks_name), "图像和掩码的数量不匹配"
        # 进一步验证图像和掩码文件名是否对应
        for img_name, mask_name in zip(self.instance_images_name, self.instance_masks_name):
            img_fname, _ = os.path.splitext(img_name)
            mask_fname, _ = os.path.splitext(mask_name)
            assert img_fname == mask_fname, f"图像文件名和掩码文件名不匹配: {img_fname} vs {mask_fname}"
            
        self.instance_prompt = instance_prompt                                      # text prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            raise ValueError(f"error. inpainting don't support prior loss.")
        else:
            self.class_data_root = None

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

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
        
        return example

    def __getitem__(self, index):
        
        image_path = os.path.join(self.instance_image_root, self.instance_images_name[index % self.num_instance_images])
        mask_path  = os.path.join(self.instance_mask_root, self.instance_masks_name[index % self.num_instance_images])
        # 读取数据
        instance_image = Image.open(image_path)
        instance_mask  = Image.open(mask_path)
 
        example = self.perform_transform(image=instance_image, mask=instance_mask)
        return example      
    



def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples] # 原图
    masked_pixel_values = [ example["instance_images"] * (example["instance_masks"] < 0.5) for example in examples]
    mask_values  = [example["instance_masks"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    masked_pixel_values = torch.stack(masked_pixel_values)
    mask_values = torch.stack(mask_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()       # images
    masked_pixel_values = masked_pixel_values.to(memory_format=torch.contiguous_format).float()       # masked_pixel_values
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()       # masks

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,         
        "pixel_values": pixel_values,   
        "masked_pixel_values": masked_pixel_values,
        "mask_values": mask_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch

class XmlInpaintingDataset(Dataset):
    # 根据xml的个数确定数据集的个数
    def __init__(   self, 
                    
                    instance_data_root,
                    instance_prompt,
                    tokenizer,                               
                    class_data_root=None,
                    class_prompt=None,
                    class_num=None,
                    size=(512, 512),
                    center_crop=False,
                    encoder_hidden_states=None,                 # text_embedding
                    class_prompt_encoder_hidden_states=None,
                    tokenizer_max_length=None,
                    exclude_tags=[],
            ):
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        
        self.xml_path = os.path.join(instance_data_root, "xml")
        self.image_path = os.path.join(instance_data_root, "images")
        
        if not os.path.exists(self.xml_path):
            raise ValueError(f"Instance {self.xml_path} xml root doesn't exists.")
        
        if not os.path.exists(self.image_path):
            raise ValueError(f"Instance {self.image_path} image root doesn't exists.")
        
        self.xml_names = sorted([
            fname for fname in os.listdir(self.xml_path)
            if fname.lower().endswith(".xml")
            ])        
        self.num_instance_xml = len(self.xml_names)                 
        
        self.image_dicts = {}
        for fname in os.listdir(self.image_path):
            if fname.lower().endswith(extensions):
                img_fname, _ = os.path.splitext(fname)
                self.image_dicts[img_fname] = fname
            
        # 进一步验证图像和掩码文件名是否对应
        # for xml_name, image_name in zip(self.xml_names, self.image_names):
        #     xml_fname, _ = os.path.splitext(xml_name)
        #     img_fname, _ = os.path.splitext(image_name)
        #     assert xml_fname == img_fname, f"xml文件名和图像文件名不匹配: {xml_fname} vs {img_fname}"
            
        self.instance_prompt = instance_prompt                                 
        self._length = self.num_instance_xml

        if class_data_root is not None:
            raise ValueError(f"error. inpainting don't support prior loss.")
        else:
            self.class_data_root = None
            
        self.exclude_tags = exclude_tags

        self.image_transforms = JointTransform(self.size, )
    
    def parse_voc_xml(self, xml_file):
        # 解析XML文件
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 创建一个字典存储解析结果
        parsed_data = {
            "filename": root.find("path").text,
            "xmlname": os.path.basename(xml_file),
            "size": {
                "width": int(root.find("size/width").text),
                "height": int(root.find("size/height").text),
                "depth": int(root.find("size/depth").text)
            },
            "objects": []
        }

        # 遍历所有object标签
        for obj in root.findall("object"):
            obj_data = {
                "name": obj.find("name").text,
                "pose": obj.find("pose").text,
                "truncated": int(obj.find("truncated").text),
                "difficult": int(obj.find("difficult").text),
                "bndbox": {
                    "xmin": float(obj.find("bndbox/xmin").text),
                    "ymin": float(obj.find("bndbox/ymin").text),
                    "xmax": float(obj.find("bndbox/xmax").text),
                    "ymax": float(obj.find("bndbox/ymax").text)
                }
            }
            parsed_data["objects"].append(obj_data)

        return parsed_data
    
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
        example["instance_images"], example["instance_masks"] = self.image_transforms(instance_image, instance_mask)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
        
        return example
    
    def get_random_wh(self, max_attempts=50):
        for _ in range(max_attempts):
            index = random.randint(0, self.num_instance_xml)
            selected_xml = os.path.join(self.xml_path, self.xml_names[index % self.num_instance_xml])
            # 解析xml文件得到目标框信息
            xml_data = self.parse_voc_xml(selected_xml)
            
            bboxes = list()
            for item in xml_data["objects"]:
                name = item["name"]
                if name in self.exclude_tags: continue
                xmin, ymin, xmax, ymax = item["bndbox"]["xmin"], item["bndbox"]["ymin"], item["bndbox"]["xmax"], item["bndbox"]["ymax"]
                bboxes.append([xmin, ymin, xmax, ymax])
            if len(bboxes) == 0: continue
            
            selected_box = random.choice(bboxes)
            x_min, y_min, x_max, y_max = selected_box
            
            
            return abs(x_max - x_min), (y_max - y_min)
        
        return 20, 15
    
    def get_input(self, img_path):
        image = Image.open(img_path)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        image_width, image_height = image.size
        box_width, box_height = self.get_random_wh()
        
        # 随机生成左上角坐标，确保框在图像范围内
        xmin = random.randint(0, image_width - box_width)
        ymin = random.randint(0, image_height - box_height)
        
        # 计算右下角坐标
        xmax = xmin + box_width
        ymax = ymin + box_height
        
        box = (xmin, ymin, xmax, ymax)
        
        # 在 mask 上绘制白色的目标框区域
        draw.rectangle(box, fill=255)
        
        return image, mask
    
        
    
    def __getitem__(self, index):
        
        xml_path = os.path.join(self.xml_path, self.xml_names[index % self.num_instance_xml])
        fname, _ = os.path.splitext(self.xml_names[index % self.num_instance_xml])
        img_path  = os.path.join(self.image_path, self.image_dicts[fname])
        
        
        # 解析xml文件得到目标框信息
        xml_data = self.parse_voc_xml(xml_path)
        
        bboxes = list()
        for item in xml_data["objects"]:
            name = item["name"]
            if name in self.exclude_tags: continue
            xmin, ymin, xmax, ymax = item["bndbox"]["xmin"], item["bndbox"]["ymin"], item["bndbox"]["xmax"], item["bndbox"]["ymax"]
            bboxes.append([xmin, ymin, xmax, ymax])
            
        # 扩边并得到image和mask
        if len(bboxes) == 0:
            return self.__getitem__(index+1)
        
        # 读取数据
        instance_image = Image.open(img_path)
        instance_image, instance_mask = self.crop_and_expand_with_masks(image=instance_image, boxes=bboxes, )
        # instance_image.save(f"datasets/zaohe/tmp/images/{self.image_dicts[fname]}")
        # instance_mask.save(f"datasets/zaohe/tmp/masks/{fname}.png")
        
        example = self.perform_transform(image=instance_image, mask=instance_mask)
        return example   



class XmlSDXLInpaintingDataset(Dataset):
    # 根据xml的个数确定数据集的个数
    def __init__(   self, 
                    instance_data_root,
                    instance_prompt,                   
                    size=(512, 512),
                    exclude_tags=[],
            ):
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        self.size = size
        
        self.xml_path = os.path.join(instance_data_root, "xml")
        self.image_path = os.path.join(instance_data_root, "images")
        
        if not os.path.exists(self.xml_path):
            raise ValueError(f"Instance {self.xml_path} xml root doesn't exists.")
        
        if not os.path.exists(self.image_path):
            raise ValueError(f"Instance {self.image_path} image root doesn't exists.")
        
        self.xml_names = sorted([
            fname for fname in os.listdir(self.xml_path)
            if fname.lower().endswith(".xml")
            ])        
        self.num_instance_xml = len(self.xml_names)                 
        
        self.image_dicts = {}
        for fname in os.listdir(self.image_path):
            if fname.lower().endswith(extensions):
                img_fname, _ = os.path.splitext(fname)
                self.image_dicts[img_fname] = fname
            
        # 进一步验证图像和掩码文件名是否对应
        # for xml_name, image_name in zip(self.xml_names, self.image_names):
        #     xml_fname, _ = os.path.splitext(xml_name)
        #     img_fname, _ = os.path.splitext(image_name)
        #     assert xml_fname == img_fname, f"xml文件名和图像文件名不匹配: {xml_fname} vs {img_fname}"
            
        self.instance_prompt = instance_prompt                                 
        self._length = self.num_instance_xml
            
        self.exclude_tags = exclude_tags

        self.image_transforms = JointTransform(self.size, )
    
    def parse_voc_xml(self, xml_file):
        # 解析XML文件
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 创建一个字典存储解析结果
        parsed_data = {
            "filename": root.find("path").text,
            "xmlname": os.path.basename(xml_file),
            "size": {
                "width": int(root.find("size/width").text),
                "height": int(root.find("size/height").text),
                "depth": int(root.find("size/depth").text)
            },
            "objects": []
        }

        # 遍历所有object标签
        for obj in root.findall("object"):
            obj_data = {
                "name": obj.find("name").text,
                "pose": obj.find("pose").text,
                "truncated": int(obj.find("truncated").text),
                "difficult": int(obj.find("difficult").text),
                "bndbox": {
                    "xmin": float(obj.find("bndbox/xmin").text),
                    "ymin": float(obj.find("bndbox/ymin").text),
                    "xmax": float(obj.find("bndbox/xmax").text),
                    "ymax": float(obj.find("bndbox/ymax").text)
                }
            }
            parsed_data["objects"].append(obj_data)

        return parsed_data
    
    def __len__(self):
        return self._length
    
    def select(self, num):
        self._length = num
        self.num_instance_xml = num
        self.xml_names = random.sample(self.xml_names, num)
        
        return self
    
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
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        box = np.int32(selected_box)
        draw.rectangle([box[0], box[1], box[2], box[3]], fill=255)  # 用白色填充目标框区域
        return image, mask
    
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
        example["instance_images"], example["instance_masks"] = self.image_transforms(instance_image, instance_mask)
        example["original_size"] = (instance_image.height, instance_image.width)
        example["crop_top_left"] = self.image_transforms.crop_top_left
        example["instance_prompt"] = self.instance_prompt

        return example
    
    def get_random_wh(self, max_attempts=50):
        for _ in range(max_attempts):
            index = random.randint(0, self.num_instance_xml)
            selected_xml = os.path.join(self.xml_path, self.xml_names[index % self.num_instance_xml])
            # 解析xml文件得到目标框信息
            xml_data = self.parse_voc_xml(selected_xml)
            
            bboxes = list()
            for item in xml_data["objects"]:
                name = item["name"]
                if name in self.exclude_tags: continue
                xmin, ymin, xmax, ymax = item["bndbox"]["xmin"], item["bndbox"]["ymin"], item["bndbox"]["xmax"], item["bndbox"]["ymax"]
                bboxes.append([xmin, ymin, xmax, ymax])
            if len(bboxes) == 0: continue
            
            selected_box = random.choice(bboxes)
            x_min, y_min, x_max, y_max = selected_box
            
            
            return abs(x_max - x_min), (y_max - y_min)
        
        return 20, 15
    
    def get_input(self, img_path):
        image = Image.open(img_path)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        image_width, image_height = image.size
        box_width, box_height = self.get_random_wh()
        
        # 随机生成左上角坐标，确保框在图像范围内
        xmin = random.randint(0, image_width - box_width)
        ymin = random.randint(0, image_height - box_height)
        
        # 计算右下角坐标
        xmax = xmin + box_width
        ymax = ymin + box_height
        
        box = (xmin, ymin, xmax, ymax)
        
        # 在 mask 上绘制白色的目标框区域
        draw.rectangle(box, fill=255)
        
        return image, mask
    
    def __getitem__(self, index):
        
        xml_path = os.path.join(self.xml_path, self.xml_names[index % self.num_instance_xml])
        fname, _ = os.path.splitext(self.xml_names[index % self.num_instance_xml])
        img_path  = os.path.join(self.image_path, self.image_dicts[fname])
        
        
        # 解析xml文件得到目标框信息
        xml_data = self.parse_voc_xml(xml_path)
        
        bboxes = list()
        for item in xml_data["objects"]:
            name = item["name"]
            if name in self.exclude_tags: continue
            xmin, ymin, xmax, ymax = item["bndbox"]["xmin"], item["bndbox"]["ymin"], item["bndbox"]["xmax"], item["bndbox"]["ymax"]
            bboxes.append([xmin, ymin, xmax, ymax])
            
        # 扩边并得到image和mask
        if len(bboxes) == 0:
            return self.__getitem__(index+1)
        
        # 读取数据
        instance_image = Image.open(img_path)
        instance_image, instance_mask = self.crop_and_expand_with_masks(image=instance_image, boxes=bboxes, )
        # instance_image.save(f"datasets/bolt_data/tmp/images/{self.image_dicts[fname]}")
        # instance_mask.save(f"datasets/bolt_data/tmp/masks/{fname}.png")
        
        example = self.perform_transform(image=instance_image, mask=instance_mask)

        return example   

def sdxl_collate_fn(examples):

    pixel_values = [example["instance_images"] for example in examples] # 原图
    masked_pixel_values = [ example["instance_images"] * (example["instance_masks"] < 0.5) for example in examples]
    mask_values  = [example["instance_masks"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]
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
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }

    return batch


class BoltSDXLInpaintingDataset(Dataset):
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
    # model_id = "weights/stabilityai/stable-diffusion-2-inpainting"
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_id,
    #     subfolder="tokenizer",
    #     revision=False,
    #     use_fast=False,
    # )
    # instance_root = "datasets/cotton_fabric"
    # train_dataset = DreamBoothInpaintingDataset(
    #     instance_data_root=instance_root,
    #     instance_prompt="a photo of a sks defeat",
    #     tokenizer=tokenizer,
    #     class_data_root=None,
    #     size=(512, 512),
    # )
    # xml_path = "datasets/zaohe/xml/train"
    # img_path = "datasets/zaohe/images/images"
    # instance_root = "datasets/bolt_data"
    instance_root = "datasets/anomaly_data_0708_0815"
    train_dataset = BoltSDXLInpaintingDataset(
        instance_root,
        instance_prompt="A close-up image of a metal hole with a missing bolt.",
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