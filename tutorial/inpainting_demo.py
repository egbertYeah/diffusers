from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler

from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.utils import load_image, make_image_grid
from diffusers.image_processor import VaeImageProcessor

device = "cuda:0"
model_id = "weights/stabilityai/stable-diffusion-2-inpainting"
# 1. 初始化各个部件
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", )
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", )
unet   = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", )

scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

print(f"scheduler: {scheduler}")

vae.to(device)
text_encoder.to(device)
unet.to(device)

# 处理文本数据得到text embedding
prompt = ["a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"]
# negative_prompt = "bad anatomy, deformed, ugly, disfigured"
height, width = 512, 512
num_inference_steps = 25  
guidance_scale = 7.5
generator = torch.cuda.manual_seed(0)  # Seed generator to create the initial latent noise
batch_size = len(prompt)

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
# generate unconditional text embeddings with same shape
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""]*batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

print(f"text embedding shape: {text_embeddings.shape}")


# load the base and mask image
init_image = load_image("docs/inpaint.png")
mask_image = load_image("docs/inpaint_mask.png")

# 下采样倍数
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
mask_processor = VaeImageProcessor(
    vae_scale_factor=vae_scale_factor,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True
)

init_image = image_processor.preprocess(
    init_image, height, width,crops_coords=None, 
)
init_image = init_image.to(dtype=torch.float32)
# 在输入图像上加噪
num_channels_latents = vae.config.latent_channels
num_channels_unet = unet.config.in_channels
# noise
shape = (
    batch_size,
    num_channels_latents,
    int(height) // vae_scale_factor,
    int(width) // vae_scale_factor,
)
# create random latent noise
latents = torch.randn(
    shape,
    generator=generator,
    device=device,
)

latents = latents * scheduler.init_noise_sigma

mask_condition = mask_processor.preprocess(
    mask_image, height, width, crops_coords=None
)
masked_image = init_image * (mask_condition < 0.5)
mask = torch.nn.functional.interpolate(
    mask_condition, size=(height // vae_scale_factor, width // vae_scale_factor)   
)
# 生成masked image latents
masked_image_latents = vae.encode(masked_image.to(device)).latent_dist.sample(generator)
masked_image_latents = vae.config.scaling_factor * masked_image_latents
mask = torch.cat([mask] * 2)
masked_image_latents = torch.cat([masked_image_latents] * 2)
mask = mask.to(device)
masked_image_latents = masked_image_latents.to(device)

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    latent_model_input = torch.cat([latents] * 2)
    
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
    
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings,).sample
    
    # perform CFG
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    
# use the vae to decode the latent representation into an image
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# convert the denoised output into an image
image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image.save("astronaut_rides_horse.png")