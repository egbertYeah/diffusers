from diffusers import StableDiffusionXLInpaintPipeline
import torch
from diffusers.utils import load_image, make_image_grid
import os
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

device = "cuda:0"
model_id = "experiments/sdxl-finetune-inpainting-bolt-512"
# unet = UNet2DConditionModel.from_pretrained("experiments/sdxl-finetune-inpainting-bolt-512/checkpoint-16000/unet",  use_safetensors=True,).half()
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_id, ).to(device)
# pipe.unet.half()

prompt = "a missing sks bolt."
generator = torch.Generator(device=device).manual_seed(0)


def expand_box( img_h, img_w, box, expansion_factor=2):
    
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

def get_input(img_path,):
    image = Image.open(img_path)
    image_width, image_height = image.size
    box_w, box_h = image_width // 3, image_height // 3
    box_x, box_y = image_width // 2, image_height // 2
    
    x_min = box_x - box_w // 2
    y_min = box_y - box_h // 2
    x_max = box_x + box_w // 2
    y_max = box_y + box_h // 2
    box = x_min, y_min, x_max, y_max

    new_xmin, new_ymin, new_xmax, new_ymax = expand_box(image_height, image_width, box)
    cropped_image = image.crop((new_xmin, new_ymin, new_xmax, new_ymax))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    # 在 mask 上绘制白色的目标框区域
    draw.rectangle(box, fill=255)
    mask = mask.crop((new_xmin, new_ymin, new_xmax, new_ymax))

    return cropped_image, mask
  
img_root = "/mnt/ssd8/xukang/datas/360_anomaly_data/box_data_right"
save_root = "datasets/tmp"
extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
for name in os.listdir(img_root):
  if not name.lower().endswith(extensions): continue
  
  img_path = os.path.join(img_root, name)
  
  image, mask = get_input(img_path, )
  init_image = image.resize((512, 512))
  init_mask = mask.resize((512, 512))
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

  # print(image.size)

  result = make_image_grid([init_image, init_mask, image], rows=1, cols=3)
  result.save(os.path.join(save_root, name))
