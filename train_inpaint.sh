#!/bin/bash
export NCCL_P2P_LEVEL=NVL

source activate
conda activate ldm

export MODEL_NAME="weights/stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="/gemini/data-1/youdamianban/2024-09-30"
export OUTPUT_DIR="experiments/sd3-controlnet-inpainting-liewen-512"

accelerate launch examples/controlnet/train_controlnet_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --variant fp16 \
  --instance_data_dir $INSTANCE_DIR \
  --instance_prompt "an image of a microelectronic circuit board with visible cracks on its surface." \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=20000 \
  --learning_rate=1e-06 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --output_dir $OUTPUT_DIR \
  --seed 10010 \
  --checkpointing_steps 2000 \
  --gradient_accumulation_steps=4 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --mixed_precision "fp16"