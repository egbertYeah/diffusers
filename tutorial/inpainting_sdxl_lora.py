"""
inpainting replaces or edits specific area of an image.
"""

from diffusers import StableDiffusionXLInpaintPipeline

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
import torch


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

    def __getitem__(self, index) :
        pass

    

    
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
        print(f"box: ({box_height, box_width})")
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
    


model_id = "weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    model_id, 
    torch_type=torch.float16,
    variant="fp16"
).to("cuda")

pipeline.enable_model_cpu_offload()
pipeline.load_lora_weights("experiments/sdxl-lora-inpainting-zaohe-512", weight_name="pytorch_lora_weights.safetensors")


instance_root = "datasets/zaohe"
train_dataset = XmlSDXLInpaintingDataset(
    instance_data_root=instance_root,
    instance_prompt="a photo of defects.",
    size=(512, 512),
)

prompt = ["a photo of defects."]
generator = torch.Generator(device="cuda").manual_seed(0)


for idx in range(500):
    init_image, mask_image = train_dataset.get_input("datasets/zaohe/tmp/template.png")

    image = pipeline(prompt=prompt, 
                    image=init_image,
                    mask_image=mask_image,
                    height=512,
                    width=512,
                    original_size=(512, 512),
                    target_size=(512, 512),
                    generator=generator,
                    num_inference_steps=20,
                    ).images[0]

    result = make_image_grid([init_image, mask_image, image], rows=1, cols=3, resize=512)


    result.save(f"datasets/zaohe/tmp/res/inpainting_sdxl_lora_{idx}.png")