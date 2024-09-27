#!/bin/bash
export NCCL_P2P_LEVEL=NVL

source activate
conda activate ldm

export MODEL_NAME="weights/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
export INSTANCE_DIR="datasets/anomaly_data_0708_0815"
export OUTPUT_DIR="experiments/sdxl-inpainting-ipadapter-bolt-512"
export CLIP_IMG_DIR="weights/h94/IP-Adapter/sdxl_models/image_encoder"

accelerate launch examples/ip_adapter/train_ip_adapter_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --image_encoder_path=${CLIP_IMG_DIR} \
  --variant fp16 \
  --train_data_dir $INSTANCE_DIR \
  --instance_prompt "a missing sks bolt." \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=10000 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=2000 \
  --output_dir $OUTPUT_DIR \
  --seed 10010 \
  --checkpointing_steps 100 \
  --prediction_type epsilon \
  --mixed_precision fp16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam 