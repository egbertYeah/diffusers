"""
inpainting replaces or edits specific area of an image.
"""

from diffusers import StableDiffusionInpaintPipeline

from diffusers.utils import load_image, make_image_grid

from torchvision import transforms

from torchvision.transforms import functional as TF

from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageOps import exif_transpose
import torch
from PIL import Image, ImageDraw
import os
import numpy as np
import random
from transformers import AutoTokenizer
import xml.etree.ElementTree as ET



class RandomResizedCropWithIoU:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), iou_threshold=0.3, max_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.iou_threshold = iou_threshold
        self.max_attempts = max_attempts

    def __call__(self, image, mask):
        for attempt in range(self.max_attempts):
            i, j, h, w = self._get_random_params(image)
            # cropped_mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)
            cropped_mask = TF.crop(mask, i, j, h, w)

            if self._compute_iou(cropped_mask, mask) >= self.iou_threshold:
                # 如果找到符合要求的裁剪框，裁剪图像和掩码
                image = TF.resized_crop(image, i, j, h, w, self.size, Image.BILINEAR)
                mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)

                return image, mask
        
        # 如果经过多次尝试未能找到合适的裁剪框，返回最后一次的裁剪结果
        image = TF.resize(image,  self.size, Image.BILINEAR)
        mask = TF.resize(mask, self.size, Image.NEAREST)
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
        
        pass


model_id = "experiments/sd-2-dreambooth-inpainting-zaohe-512"
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    model_id, 
)

pipeline.enable_model_cpu_offload()


tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    subfolder="tokenizer",
    revision=False,
    use_fast=False,
)

instance_root = "datasets/zaohe"
train_dataset = XmlInpaintingDataset(
    instance_root,
    instance_prompt="a photo of a sks defects.",
    tokenizer=tokenizer,
    class_data_root=None,
    size=(512, 512),
)
# load the base and mask image
# init_image = load_image("datasets/cotton_fabric/images/00969_0013_Harness-Misdraw.jpg")
# mask_image = load_image("datasets/cotton_fabric/masks/00963_0006_Slubs.png")

init_image, mask_image = train_dataset.get_input("datasets/zaohe/tmp/images/09-45-41-584_2.jpg")
# transform = RandomResizedCropWithIoU(size=(512, 512))
# init_image, mask_image = transform(init_image, mask_image)
# 对mask进行滤波处理, 使得原图和inpaint区域不那么突兀
# blurred_mask = pipeline.mask_processor.blur(mask_image, blur_factor=3)

prompt = ["a photo of sks defects."]
# negative_prompt = "bad anatomy, deformed, ugly, disfigured"
# padding_mask_crop: 首先获取最小外接矩形, 然后按照padding_mask_crop对区域进行扩边, 考虑输入的宽高比, 将区域调整到对应的宽高比, 最后resize到网络输入大小
# image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=blurred_mask, padding_mask_crop=32).images[0]

image = pipeline(prompt=prompt, image=init_image,mask_image=mask_image,).images[0]

result = make_image_grid([init_image, mask_image, image], rows=1, cols=3, resize=512)


result.save("inpainting.png")