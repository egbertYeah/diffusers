from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler

from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
import torch
from tqdm.auto import tqdm

device = "cuda:0"
model_id = "weights/stabilityai/stable-diffusion-2-1-base"
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


# 2. tokenize the text to generate embeddings
prompt = ["Close-up of a metal surface with a circular hole where a bolt is missing, showing signs of wear and scratches around the hole. The lighting highlights the texture of the metal, with visible scratches and abrasions caused by previous bolt replacements or removals. To the right side of the image, there is a sharp edge or boundary of another object, suggesting part of a mechanical component or machinery. The image should have a grayscale or monochrome effect, emphasizing the details and textures of the metal surface."]
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

# create random latent noise
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=device,
)

# denoise the image
# for improved schedulers, the input scaled by the initial noise distribution
latents = latents * scheduler.init_noise_sigma

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    latent_model_input = torch.cat([latents] * 2)
    
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
    
    # predict the moise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    
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