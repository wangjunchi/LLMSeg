#! /bin/bash


llava_path="./pretrained_weights/LLaVA-lightning-7B-v1/"
vision_path="./pretrained_weights/SAM/sam_vit_h_4b8939.pth"
dataset_path="./lisa_dataset"
sam_masks_path="./processed_data"
log_path="./runs"

deepspeed --include localhost:0,1 \
  --master_port=24371 training_debug.py \
  --version="$llava_path" \
  --dataset_dir="$dataset_path" \
  --sam_masks_dir="$sam_masks_path" \
  --vision_pretrained="$vision_path" \
  --dataset="sem_seg||refer_seg" \
  --sample_rates="9,3" \
  --exp_name="zeroshot" \
  --log_base_dir="$log_path" \
  --lr=0.0001 \
  --epochs=10 \
  --batch_size=1 \
