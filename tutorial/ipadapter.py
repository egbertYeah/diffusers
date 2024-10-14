from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image, make_image_grid
import torch


model_id = "weights/stabilityai/stable-diffusion-xl-base-1.0"
ipadapter_id = "weights/h94/IP-Adapter"
image_path = "tutorial/ip_adapter_diner.png"
gpu_id = 0
device = f"cuda:{gpu_id}"
prompt = "a polar bear sitting in a chair drinking a milkshake"
negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"


pipeline = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to(device)
print("loaded pipeline.")
pipeline.load_ip_adapter(ipadapter_id, subfolder="sdxl_models", weight_name="ip-adapter_sdxl_vit-h.bin")
pipeline.set_ip_adapter_scale(0.6)
print("loaded ip-adapter.")
ipadapter_image = load_image(image_path)
generator = torch.Generator(device=device).manual_seed(0)

result = pipeline(
    prompt=prompt,
    ip_adapter_image=ipadapter_image,
    negative_prompt=negative_prompt,
    num_inference_steps=100,
    generator=generator,
).images[0]

grid_result = make_image_grid([ipadapter_image, result], rows=1, cols=2)
grid_result.save("ipadapter_txt2img.png")