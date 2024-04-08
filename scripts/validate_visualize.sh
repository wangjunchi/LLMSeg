#! /bin/bash


llava_path="./pretrained_weights/LLaVA-lightning-7B-v1/"
vision_path="./pretrained_weights/SAM/sam_vit_h_4b8939.pth"
dataset_path="./lisa_dataset"
sam_masks_path="./processed_data"
log_path="./runs"


deepspeed --include localhost:2,3 \
  --master_port=24353 training_debug.py \
  --version="$llava_path" \
  --dataset_dir="$dataset_path" \
  --sam_masks_dir="$sam_masks_path" \
  --vision_pretrained="$vision_path" \
  --dataset="reason_seg" \
  --sample_rates="1" \
  --exp_name="10epoch" \
  --log_base_dir="$log_path" \
  --batch_size=1 \
  --eval_only \
  --val_dataset="ReasonSeg|val" \
  --visualize