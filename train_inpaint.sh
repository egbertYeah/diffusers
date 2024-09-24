#!/bin/bash
export NCCL_P2P_LEVEL=NVL

source activate
conda activate ldm

export MODEL_NAME="weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
export INSTANCE_DIR="datasets/anomaly_data_0708_0815"
export OUTPUT_DIR="experiments/sdxl-finetune-inpainting-bolt-512"

accelerate launch examples/dreambooth-inpainting/train_sdxl_inpainting.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --variant fp16 \
  --train_data_dir $INSTANCE_DIR \
  --instance_prompt "a missing sks bolt." \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=20000 \
  --learning_rate=1e-06 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --output_dir $OUTPUT_DIR \
  --seed 10010 \
  --checkpointing_steps 2000 \
  --prediction_type epsilon \
  --mixed_precision fp16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam 