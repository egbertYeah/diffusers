import torch
from diffusers.utils import load_image, check_min_version, make_image_grid
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models.controlnet_sd3 import SD3ControlNetModel
from transformers import T5EncoderModel, BitsAndBytesConfig


controlnet = SD3ControlNetModel.from_pretrained(
    "weights/alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1
)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "weights/stabilityai/stable-diffusion-3-medium-diffusers"
text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    use_safetensors=True,variant="fp16",
    quantization_config=quantization_config,
)
pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    text_encoder_3=text_encoder,
    torch_dtype=torch.float16,
    device_map="balanced",
    use_safetensors=True,variant="fp16"
)
pipe.text_encoder.to(torch.float16)
pipe.controlnet.to(torch.float16)
pipe.to("cuda")
# pipe.enable_model_cpu_offload()

image = load_image(
    "weights/alimama-creative/SD3-Controlnet-Inpainting/images/dog.png"
).resize((1024, 1024))
mask = load_image(
    "weights/alimama-creative/SD3-Controlnet-Inpainting/images/dog_mask.png"
).resize((1024, 1024))
width = 1024
height = 1024
prompt = "A cat is sitting next to a puppy."
# generator = torch.Generator(device="cuda").manual_seed(24)
res_image = pipe(
    negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
    prompt=prompt,
    height=height,
    width=width,
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    # generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=7,
).images[0]

result = make_image_grid([image, mask, res_image], rows=1, cols=3)

result.save(f"sd3.png")
