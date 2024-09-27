from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import torch
from ip_adapter.ip_adapter import IPAdapterXLInpainting
import pdb

model_id = "weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
ip_ckpt = "experiments/sdxl-inpainting-ipadapter-bolt-512/checkpoint-10000/model.safetensors"
image_encoder_folder = "weights/h94/IP-Adapter/sdxl_models/image_encoder"
gpu_id = 0
device = f"cuda:{gpu_id}"
prompt = "a missing sks bolt."
# negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"
generator = torch.Generator(device=device).manual_seed(4)




pipeline = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
print("loaded pipeline.")

# pdb.set_trace()
# load ip-adapter
ip_model = IPAdapterXLInpainting(pipeline, image_encoder_folder, ip_ckpt, device)
# pdb.set_trace()

while True:
    command    = input("请输入命令, q表示退出, r表示继续: ")
    if command == "q":
        break
    if command == "r":
        pass
    image_path = input("请输入图像路径: ")
    mask_path  = input("请输入mask路径: ")
    ipa_path   = input("请输入IPA路径: ")
    save_path  = input("请输入结果保存路径: ")
    mask_image = load_image(mask_path).resize((512, 512))
    image = load_image(image_path).resize((512, 512))
    ip_image = load_image(ipa_path).resize((512, 512))

    result = ip_model.generate(
        image=image,
        mask_image=mask_image,
        ip_adapter_image=ip_image,
        prompt=prompt,
        height=512, width=512,
        original_size=(512, 512),
        target_size=(512, 512),
        ipa_scale=0.6,
        # negative_prompt=negative_prompt,
        num_inference_steps=100,
        generator=generator,
    )[0]

    grid_result = make_image_grid([image, ip_image, result], rows=1, cols=3)
    grid_result.save(save_path)