from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import torch


model_id = "weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
ipadapter_id = "weights/h94/IP-Adapter"

gpu_id = 0
device = f"cuda:{gpu_id}"
prompt = "a cute gummy bear waving"
# negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"

mask_image = load_image("examples/ip_adapter/ip_adapter_mask.png").resize((1024, 768))
image = load_image("examples/ip_adapter/ip_adapter_bear_1.png").resize((1024, 768))
ip_image = load_image("examples/ip_adapter/ip_adapter_gummy.png").resize((1024, 768))
generator = torch.Generator(device=device).manual_seed(4)



pipeline = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
print("loaded pipeline.")
pipeline.load_ip_adapter(ipadapter_id, subfolder="sdxl_models", weight_name="ip-adapter_sdxl_vit-h.bin")
pipeline.set_ip_adapter_scale(0.6)
print("loaded ip-adapter.")



result = pipeline(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    ip_adapter_image=ip_image,
    height=768, width=1024,
    original_size=(1024, 768),
    target_size=(1024, 768),
    # negative_prompt=negative_prompt,
    num_inference_steps=100,
    generator=generator,
).images[0]

grid_result = make_image_grid([image, ip_image, result], rows=1, cols=3)
grid_result.save("ipadapter_inpainting.png")