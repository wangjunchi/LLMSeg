'''
COCO training set: 118,287 images, if we naively use 1 GPU to process 1 image, it will take almost a week to process the whole dataset.

A more efficient and simple way is to split the dataset into 8 parts, and we can submit 8 jobs to the cluster to process the dataset in parallel.
Ideally, it will take 1 day to process the whole dataset.

'''

import os
import numpy as np
from typing import List
import cv2
import h5py
import argparse
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# accept split number as argument
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=int, required=True, help="split number")
parser.add_argument("--dataset_path", type=str, required=True, help="path to coco dataset")
parser.add_argument("--split_dir", type=str, required=True, help="path to split directory")
parser.add_argument("--sam_checkpoint", type=str, default="/cluster/home/leikel/junchi/segment-anything/checkpoints/sam_vit_h_4b8939.pth",
                     help="path to sam checkpoint")

def preprocess_images(image: np.ndarray) -> np.ndarray:
    # scale the large side to 1024
    H, W, _ = image.shape
    if max(W, H) > 1024:
        # scale
        scale_factor = 1024.0 / max(W, H)
        image = cv2.resize(image, (int(W*scale_factor), int(H*scale_factor)), interpolation = cv2.INTER_AREA)

    return image


def init_SAM_everything(model_type: str, sam_checkpoint: str) -> SamAutomaticMaskGenerator:
    # check if cuda is available
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


def read_split_file(split_file: str) -> List[str]:
    with open(split_file, "r") as f:
        lines = f.readlines()
    images = [line.strip() for line in lines]
    # check if it is a valid image fil
    for image in images:
        assert image.endswith(".jpg"), "Invalid image file: {}".format(image)
        
    return images


def process_split(split_file: str, dataset_path: str, sam_checkpoint: str, args) -> None:
    # read split file
    images = read_split_file(split_file)

    split_num = int(split_file.split("_")[-1].split(".")[0])
    print("Processing split {} with {} images".format(split_num, len(images)))
    
    # init SAM
    mask_generator = init_SAM_everything("vit_h", sam_checkpoint)
    
    masks_all_samples = []

    # process images
    for idx, image_file in enumerate(images):
        # read image
        image_path = os.path.join(dataset_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_images(image)
        
        # generate mask
        masks = mask_generator.generate(image)
        
        sample_dict = {}

        sample_dict["image"] = image_file
        sample_dict["target_size"] = [image.shape[0], image.shape[1]]

        # convert masks to coco format
        masks_coco = []
        for mask in masks:
            binary_mask = mask['segmentation']
            mask_rel = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
            mask['segmentation'] = mask_rel
            masks_coco.append(mask)

        sample_dict["masks"] = masks_coco

        masks_all_samples.append(sample_dict)

        
        if idx % 10 == 0:
            print("Processing image {}/{}".format(idx, len(images)))

        # # debug
        # if idx == 10:
        #     break
        
    h5_save_path = os.path.join(args.split_dir, "coco_split{}.h5".format(split_num))

    # convert dict to string
    # https://stackoverflow.com/questions/16494669/how-to-store-dictionary-in-hdf5-dataset
    masks_all_samples_str = []
    for sample in masks_all_samples:
        masks_all_samples_str.append(str(sample))

    with h5py.File(h5_save_path, 'w') as f:
        f.create_dataset('masks', data=masks_all_samples_str)


def main():
    args = parser.parse_args()
    split_num = args.split
    dataset_path = args.dataset_path
    split_dir = args.split_dir
    sam_checkpoint = args.sam_checkpoint

    assert split_num >= 0 and split_num < 8, "Invalid split number: {}".format(split_num)
    
    split_file = os.path.join(split_dir, "part_{}.txt".format(split_num))
    process_split(split_file, dataset_path, sam_checkpoint, args)

    print("Done.")


if __name__ == "__main__":
    main()