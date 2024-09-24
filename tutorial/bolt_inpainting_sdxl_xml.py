from diffusers import StableDiffusionXLInpaintPipeline
import torch
from diffusers.utils import load_image, make_image_grid
import os
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import xml.etree.ElementTree as ET



def parse_voc_xml( xml_file):
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
      
device = "cuda:0"
model_id = "experiments/sdxl-finetune-inpainting-bolt-512"
# unet = UNet2DConditionModel.from_pretrained("experiments/sdxl-finetune-inpainting-bolt-512/checkpoint-16000/unet",  use_safetensors=True,).half()
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_id, ).to(device)
# pipe.unet.half()

prompt = "A close-up image of a metal hole with a missing sks bolt."
generator = torch.Generator(device=device).manual_seed(0)


instance_data_root = "datasets/normal_data"
xml_root = os.path.join(instance_data_root, "xml")
img_root = os.path.join(instance_data_root, "images")
save_root = "datasets/normal_data/res"
extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

for xml_name in os.listdir(xml_root):
  xml_path = os.path.join(xml_root, xml_name)
  
  fname, _ = os.path.splitext(xml_name)
  
  img_path  = os.path.join(img_root, fname+".png")
  
  image = Image.open(img_path).convert("RGB")
  # 解析xml文件得到目标框信息
  xml_data = parse_voc_xml(xml_path)
  
  for item in xml_data["objects"]:
      name = item["name"]
      xmin, ymin, xmax, ymax = item["bndbox"]["xmin"], item["bndbox"]["ymin"], item["bndbox"]["xmax"], item["bndbox"]["ymax"]
      box = [xmin, ymin, xmax, ymax]
      
      mask = Image.new("L", image.size, 0)
      draw = ImageDraw.Draw(mask)
      draw.rectangle([box[0], box[1], box[2], box[3]], fill=255)  # 用白色填充目标框区域
            

      init_image = image.resize((512, 512))
      init_mask = mask.resize((512, 512))
      # print(init_image.size)
      # print(init_mask.size)
      image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=init_mask,
        height=512,
        width=512,
        original_size=(512, 512),
        target_size=(512, 512),
        guidance_scale=7.0,
        num_inference_steps=30,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
      ).images[0]

      print(image.size)

      result = make_image_grid([init_image, init_mask, image], rows=1, cols=3)
      result.save(os.path.join(save_root, fname+".png"))
