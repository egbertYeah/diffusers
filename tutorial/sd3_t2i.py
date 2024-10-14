import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import SD3Transformer2DModel


model_id = "weights/stabilityai/stable-diffusion-3-medium-diffusers"
transformer = SD3Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16, variant="fp16")
print(transformer)
# pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", transformer=transformer)

# pipe.enable_model_cpu_offload()

# image = pipe(
#     "A cat holding a sign that says hello world",
#     negative_prompt="",
#     num_inference_steps=28,
#     guidance_scale=7.0,
# ).images[0]

# image.save("sd3_hello_world.png")
