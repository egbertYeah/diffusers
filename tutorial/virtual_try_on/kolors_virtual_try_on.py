from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import StableDiffusionXLInpaintPipeline
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor

# from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import  AutoencoderKL
from kolors.models.unet_2d_condition import UNet2DConditionModel

from diffusers import EulerDiscreteScheduler
from diffusers.utils import load_image, make_image_grid
import torch
from SegBody import segment_body


ckpt_dir = f'weights/Kwai-Kolors/Kolors-Inpainting'
ipadapter_id = "weights/Kwai-Kolors/Kolors-IP-Adapter-Plus/"

gpu_id = 0
device = f"cuda:{gpu_id}"
prompt = "photorealistic, perfect body, beautiful skin, realistic skin, natural skin"
negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings"
# negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"

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

image_encoder = CLIPVisionModelWithProjection.from_pretrained( f'weights/Kwai-Kolors/Kolors-IP-Adapter-Plus/image_encoder',  ignore_mismatched_sizes=True).to(dtype=torch.float16)
ip_img_size = 336
clip_image_processor = CLIPImageProcessor( size=ip_img_size, crop_size=ip_img_size )

print(f"loaded image encoder and processor.")


pipeline = StableDiffusionXLInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        image_encoder=image_encoder,
        feature_extractor=clip_image_processor,
        force_zeros_for_empty_prompt=False,
        add_watermarker=False
)

pipeline.to(device)
pipeline.enable_attention_slicing()
print("loaded pipeline.")
if hasattr(pipeline.unet, 'encoder_hid_proj'):
    pipeline.unet.text_encoder_hid_proj = pipeline.unet.encoder_hid_proj
    

pipeline.load_ip_adapter(ipadapter_id, subfolder="", weight_name="ip_adapter_plus_general.bin")
pipeline.set_ip_adapter_scale(1.0)
print("loaded ip-adapter.")

generator = torch.Generator(device=device).manual_seed(4)

while True:
    command    = input("请输入命令, q表示退出, r表示继续: ")
    if command == "q":
        break
    if command == "r":
        pass
    image_path = input("请输入图像路径: ")
    ipa_path   = input("请输入IPA路径: ")
    save_path  = input("请输入保存路径: ")

    image = load_image(image_path).convert("RGB").resize((1024, 1024))
    seg_image, mask_image = segment_body(image, face=False)
    ip_image = load_image(ipa_path).convert("RGB").resize((1024, 1024))


    final_image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        ip_adapter_image=ip_image,
        strength=0.99,
        guidance_scale=5.0,
        num_inference_steps=50, 
        generator=generator,
    ).images[0]

    grid_result = make_image_grid([image, ip_image, final_image], rows=1, cols=3)
    grid_result.save(save_path)