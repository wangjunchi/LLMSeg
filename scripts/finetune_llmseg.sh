#! /bin/bash


llava_path="./pretrained_weights/LLaVA-lightning-7B-v1/"
vision_path="./pretrained_weights/SAM/sam_vit_h_4b8939.pth"
dataset_path="./lisa_dataset"
sam_masks_path="./processed_data"
log_path="./runs"
resume_path="./runs/10epoch/ckpt_model"

deepspeed --include localhost:2,3 \
  --master_port=24374 finetune_llmseg.py \
  --version="$llava_path" \
  --dataset_dir="$dataset_path" \
  --sam_masks_dir="$sam_masks_path" \
  --vision_pretrained="$vision_path" \
  --dataset="sem_seg||refer_seg||reason_seg" \
  --sample_rates="9,3,1" \
  --exp_name="finetune_llmseg" \
  --log_base_dir="$log" \
  --steps_per_epoch=500 \
  --lr=1e-5 \
  --epochs=5 \
  --batch_size=1 \
  --resume='$resume_path' \
