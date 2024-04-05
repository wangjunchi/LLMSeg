#! /bin/bash


llava_path="/home/leikel/junchi/pretrained_weights/LLaVA-lightning-7B-v1/"
sam_path="/home/leikel/junchi/pretrained_weights/SAM/sam_vit_h_4b8939.pth"
dataset_path="/home/leikel/junchi/lisa_dataset"
sam_masks_path="/home/leikel/junchi/processed_data"
log_path="/home/leikel/junchi/lisa_dataset/new_runs"

deepspeed --include localhost:6,7 \
  --master_port=24374 training.py \
  --version="$llava_path" \
  --dataset_dir="$dataset_path" \
  --sam_masks_dir="$sam_masks_path" \
  --vision_pretrained="$sam_path" \
  --dataset="sem_seg||refer_seg||reason_seg" \
  --sample_rates="9,3,1" \
  --exp_name="10epoch" \
  --log_base_dir="$log_path" \
  --lr=0.0001 \
  --epochs=10 \
  --batch_size=1 \
