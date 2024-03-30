# LLM-Seg: Bridging Image Segmentation and Large Language Model Reasoning

This is the official repository for the paper "LLM-Seg: Bridging Image Segmentation and Large Language Model Reasoning" (CVPR Workshop 2024).

Our project is based on [LISA](https://github.com/dvlab-research/LISA). We thank the authors for their great work.

## Overview
LLM-Seg is a reasoning segmentation model that combines SAM and LLaVA.

### Reasoning Segmentation
![image](imgs/reasonseg_overview.png)

### Model Architecture
![image](imgs/llmseg_overview.png)


## Experiment Results
The table below shows the performance of LLM-Seg on ReasonSeg validation set.
![image](imgs/llmseg_exp.png)

![image](imgs/reasonseg_results_final_small.drawio.png)

## Prepare the environment
We recommend using conda to create a virtual environment and install the dependencies.
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```


## Preparing the dataset
Please first refer to the [LISA](https://github.com/dvlab-research/LISA) repository to download all the datasets.

After downloading the dataset, you can use the python script from `prepare_datasets.py` to preprocess the different dataset. The script will extract SAM Everything masks and save them as h5 files.

```bash
python prepare_datasets/prepare_<dataset_name>.py
```

After preprocessing all datasets, run the following script to convert the h5 files to json files.

```bash
python prepare_datasets/convert_h5_to_json.py
```

## Prepare the pretrained models
Please refer to [LISA](https://github.com/dvlab-research/LISA) repository to download the pretrained models. For LLaVA, please download the `LLaVA-lightning-7B-v1` checkpoint.

## Training the model
To train the model, create a bash script with the following content and run it.

```bash
deepspeed --include localhost:0,1 \
  --master_port=24374 training.py \
  --version=<path of LLaVA-lightning-7B-v1 checkpoint> \
  --dataset_dir=<path of datasets> \
  --sam_masks_dir=<path of SAM generated masks> \
  --vision_pretrained=<path for SAM checkpoint> \
  --dataset="sem_seg||refer_seg||reason_seg" \
  --sample_rates="9,3,1" \
  --exp_name="llmseg-20peoch" \
  --log_base_dir=<path of saved model> \
  --lr=0.0001 \
  --epochs=20 \
  --batch_size=1 \
```

## Evaluation
To evaluate the trained model, create a bash script with the following content and run it.

```bash
deepspeed --include localhost:0,1 \
  --master_port=24353 training_debug.py \
  --version=<path of LLaVA-lightning-7B-v1 checkpoint> \
  --dataset_dir=<path of datasets> \
  --sam_masks_dir=<path of SAM generated masks> \
  --vision_pretrained=<path for SAM checkpoint> \
  --dataset="reason_seg" \
  --sample_rates="1" \
  --exp_name="llmseg-20peoch" \
  --log_base_dir=<path of saved model> \
  --batch_size=1 \
  --eval_only \
  --val_dataset="ReasonSeg|val"
```

We also provide the trained checkpoint for evaluation. You can download it from huggingface. Please note the chekcpoint is in Deepspeed format, not huggingface format.

Chekcpoint for 20 epochs: [llmseg-20epoch](https://huggingface.co/JCdesu/LLM-Seg-deepspeed)

Chekcpoint for 10 epochs: [llmseg-10epoch](https://huggingface.co/JCdesu/LLM-Seg-deepspeed-10epoch)


## Acknowledgement
Our project is based on the following repositories:
- [LISA](https://github.com/dvlab-research/LISA)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [DINOv2](https://github.com/facebookresearch/dinov2)

We thank the authors for their great work. Please refer to their repositories for more details.
