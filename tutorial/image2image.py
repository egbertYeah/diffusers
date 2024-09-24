"""
image-to-image 
input: text, image
text通过text_encoder得到embedding, image通过vae的encoder得到 noisy latent image

text embedding、noisy latent image和timestamp输入到model中预测噪声

"""

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
model_id = "weights/stabilityai/stable-diffusion-2-1-base"


pipeline = AutoPipelineForImage2Image.from_pretrained(
    model_id, 
)
pipeline.enable_model_cpu_offload()

init_image = load_image("docs/cat.png")

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image, strength=1.0, guidance_scale=7.5).images[0]
result = make_image_grid([init_image, image], rows=1, cols=2)

result.save("img2img.png")