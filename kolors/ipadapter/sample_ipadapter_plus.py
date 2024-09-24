import torch
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from diffusers.utils import load_image
import os,sys

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer

# from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import  AutoencoderKL
from kolors.models.unet_2d_condition import UNet2DConditionModel

from diffusers import EulerDiscreteScheduler
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def infer( ip_img_path, prompt ):

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
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained( f'weights/Kwai-Kolors/Kolors-IP-Adapter-Plus/image_encoder',  ignore_mismatched_sizes=True).to(dtype=torch.float16)
    ip_img_size = 336
    clip_image_processor = CLIPImageProcessor( size=ip_img_size, crop_size=ip_img_size )

    pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=clip_image_processor,
            force_zeros_for_empty_prompt=False
            )

    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    if hasattr(pipe.unet, 'encoder_hid_proj'):
        pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj
    
    pipe.load_ip_adapter( f'weights/Kwai-Kolors/Kolors-IP-Adapter-Plus' , subfolder="", weight_name=["ip_adapter_plus_general.bin"])

    basename = ip_img_path.rsplit('/',1)[-1].rsplit('.',1)[0]
    ip_adapter_img = Image.open( ip_img_path )
    generator = torch.Generator(device="cpu").manual_seed(66)
    
    for scale in [0.5]:
        pipe.set_ip_adapter_scale([ scale ])
        # print(prompt)
        image = pipe(
            prompt= prompt ,
            ip_adapter_image=[ ip_adapter_img ],
            negative_prompt="", 
            height=1024,
            width=1024,
            num_inference_steps= 50, 
            guidance_scale=5.0,
            num_images_per_prompt=1,
            generator=generator,
        ).images[0]
        image.save(f'ipadapter.png')


if __name__ == '__main__':
    # import fire
    # fire.Fire(infer)
    ip_img_path = "kolors/ipadapter/asset/1.png"
    prompt = "一个看向远山的少女形象，雪山背景，采用日本浮世绘风格，混合蓝色和红色柔和调色板，高分辨率"
    infer( ip_img_path, prompt )
    
    









