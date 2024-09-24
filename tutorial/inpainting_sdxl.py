from diffusers import AutoPipelineForInpainting
import torch
from diffusers.utils import load_image, make_image_grid
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from freeu import register_free_upblock2d, register_free_crossattn_upblock2d

device = "cuda:0"
model_id = "weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

pipe = AutoPipelineForInpainting.from_pretrained("weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

register_free_upblock2d(pipe, b1=1.2, b2=1.4, s1=0.9, s2=0.9)
register_free_crossattn_upblock2d(pipe, b1=1.2, b2=1.4, s1=0.9, s2=0.9)
print(f"use freeU.")
img_url = "weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/overture-creations-5sI6fQgYIuo.png"
mask_url = "weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((1024, 1024))
init_mask = load_image(mask_url).resize((1024, 1024))

prompt = "a Golden Retriever sitting on a park bench"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=init_image,
  mask_image=init_mask,
  height=1024,
  width=1024,
  original_size=(1024, 1024),
  target_size=(1024, 1024),
  guidance_scale=7.0,
  num_inference_steps=30,  # steps between 15 and 30 work well for us
  # strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]

print(image.size)

result = make_image_grid([init_image, init_mask, image], rows=1, cols=3)
result.save("inpainting_sdxl_freeu.png")
