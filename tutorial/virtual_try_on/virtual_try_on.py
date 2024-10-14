from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import torch
from SegBody import segment_body


model_id = "weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
ipadapter_id = "weights/h94/IP-Adapter"

gpu_id = 0
device = f"cuda:{gpu_id}"
prompt = "photorealistic, perfect body, beautiful skin, realistic skin, natural skin"
negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings"
# negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"


pipeline = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to(device)
print("loaded pipeline.")
pipeline.load_ip_adapter(ipadapter_id, subfolder="sdxl_models", weight_name="ip-adapter_sdxl_vit-h.bin")
pipeline.set_ip_adapter_scale(1.0)
print("loaded ip-adapter.")

image = load_image('tutorial/virtual_try_on/jpFBKqYB3BtAW26jCGJKL.jpeg').convert("RGB")
seg_image, mask_image = segment_body(image, face=False)
ip_image = load_image('tutorial/virtual_try_on/NL6mAYJTuylw373ae3g-Z.jpeg').convert("RGB")
generator = torch.Generator(device=device).manual_seed(4)


final_image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask_image,
    ip_adapter_image=ip_image,
    strength=0.99,
    guidance_scale=7.5,
    num_inference_steps=100,
    generator=generator,
).images[0]

grid_result = make_image_grid([image, ip_image, final_image], rows=1, cols=3)
grid_result.save("ipadapter_inpainting.png")