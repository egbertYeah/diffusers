from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
import torch

original_image = load_image(
    "tutorial/controlnet/hf-logo.png"
)

model_id = "weights/stabilityai/stable-diffusion-xl-base-1.0"
controlnet_id = "weights/diffusers/controlnet-canny-sdxl-1.0"

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
print("loaded controlnet.")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
print("loaded pipeline.")
pipe.enable_model_cpu_offload()


prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    controlnet_conditioning_scale=0.5,
).images[0]
grid_result = make_image_grid([original_image, canny_image, image], rows=1, cols=3)

grid_result.save("tutorial/controlnet/res.png")
