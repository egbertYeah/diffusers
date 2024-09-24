import os, torch
# from PIL import Image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def infer(prompt):
    ckpt_dir = f'weights/Kwai-Kolors/Kolors-diffusers'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder', ).half()
    print("loaded text encoder.")
    
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", ).half()
    print("loaded VAE.")
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).half()
    print("loaded UNet.")
    
    pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False)
    
    pipe = pipe.to("cuda")
    print("loaded pipeline to GPU.")
    pipe.enable_model_cpu_offload()
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator= torch.Generator(pipe.device).manual_seed(66)).images[0]
    image.save("1_prompt.png")


if __name__ == '__main__':
    # import fire
    # fire.Fire(infer)

    prompt = "一对年轻的中国情侣，皮肤白皙，穿着时尚的运动装，背景是现代的北京城市天际线。面部细节，清晰的毛孔，使用最新款的相机拍摄，特写镜头，超高画质，8K，视觉盛宴"
    infer(prompt)

