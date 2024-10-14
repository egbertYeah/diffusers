from diffusers import AutoPipelineForInpainting
import torch
from diffusers.utils import load_image, make_image_grid
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionXLInpaintPipeline

device = "cuda:0"
model_id = "experiments/sdxl-finetune-inpainting-liewen-512"

# unet   = UNet2DConditionModel.from_pretrained("experiments/sdxl-finetune-inpainting-liewen-512/checkpoint-10000/unet", )

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_id, ).to(device)

prompt = "an image of a microelectronic circuit board with visible cracks on its surface."
generator = torch.Generator(device=device).manual_seed(0)


while True:
    command    = input("请输入命令, q表示退出, r表示继续: ")
    if command == "q":
        break
    if command == "r":
        pass
    image_path = input("请输入图像路径: ")
    mask_path  = input("请输入mask路径: ")
    save_path  = input("请输入结果保存路径: ")

    init_image = load_image(image_path).resize((512, 512))
    init_mask = load_image(mask_path).resize((512, 512))

    image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=init_mask,
    height=512,
    width=512,
    original_size=(512, 512),
    target_size=(512, 512),
    guidance_scale=7.0,
    num_inference_steps=30,  # steps between 15 and 30 work well for us
    strength=0.99,  # make sure to use `strength` below 1.0
    generator=generator,
    ).images[0]


    grid_result = make_image_grid([init_image, init_mask, image], rows=1, cols=3)
    grid_result.save(save_path)

