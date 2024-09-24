import torch
from diffusers import StableDiffusion3InpaintPipeline, StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from diffusers.utils import load_image, make_image_grid


model_id = "weights/stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3InpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True,variant="fp16")
pipe.to("cuda")
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
img_url = "docs/overture-creations-5sI6fQgYIuo.png"
mask_url = "docs/overture-creations-5sI6fQgYIuo_mask.png"
source = load_image(img_url).resize((1024, 1024))
mask = load_image(mask_url).resize((1024, 1024))

image = pipe(prompt=prompt, image=source, mask_image=mask).images[0]

result = make_image_grid([source, mask, image], rows=1, cols=3)

result.save("sd3_inpainting.png")