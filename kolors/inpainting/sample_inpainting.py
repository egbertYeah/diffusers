import torch
import os, sys
from PIL import Image

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import StableDiffusionXLInpaintPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer

def infer(image_path, mask_path, prompt):

    ckpt_dir = f'weights/Kwai-Kolors/Kolors-Inpainting'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    print(f"loaded text encoder.")
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    print(f"loaded vae.")
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()
    print(f"loaded unet.")

    pipe = StableDiffusionXLInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            add_watermarker=False
    )
    
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cuda").manual_seed(603)
    # basename = image_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    image = Image.open(image_path).convert('RGB')
    # mask_image = Image.open(mask_path).convert('RGB')
    mask_image = Image.open(mask_path).convert('L')

    result = pipe(
        prompt = prompt,
        image = image,
        mask_image = mask_image,
        height=1024,
        width=768,
        guidance_scale = 6.0,
        generator= generator,
        num_inference_steps= 25,
        negative_prompt = '残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量',
        num_images_per_prompt = 1,
        strength = 0.999
    ).images[0]
    result.save(f'4_inpainting.png')

if __name__ == '__main__':
    # import fire
    # fire.Fire(infer)
    image_path = "kolors/inpainting/asset/4.png"
    mask_path = "kolors/inpainting/asset/4_mask.png"
    prompt = "穿着钢铁侠的衣服，高科技盔甲，主要颜色为红色和金色，并且有一些银色装饰。胸前有一个亮起的圆形反应堆装置，充满了未来科技感。超清晰，高质量，超逼真，高分辨率，最好的质量，超级细节，景深"
    infer(image_path, mask_path, prompt)
