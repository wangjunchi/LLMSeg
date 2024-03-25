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
parser.add_argument("--input_path", type=str, default="/cluster/scratch/leikel/junchi/lisa_dataset/ade20k/images/training",
                     help="path to coco dataset")
parser.add_argument("--output_path", type=str, default="/cluster/home/leikel/junchi/processed_data/ade20k", 
                    help="path to split directory")
parser.add_argument("--sam_checkpoint", type=str, default="/cluster/home/leikel/junchi/segment-anything/checkpoints/sam_vit_h_4b8939.pth",
                     help="path to sam checkpoint")

def get_all_samples(dataset_path) -> List[str]:
    
    all_samples = []
    files = os.listdir(dataset_path)

    for file in files:
        if file.endswith(".jpg"):
            name = file.split('.')[0]
            all_samples.append(name)

    return all_samples


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


def process_dataset(input_path: str, output_path: str, sam_checkpoint: str) -> None:

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    sample_list = get_all_samples(input_path)

    mask_generator = init_SAM_everything(model_type="vit_h", sam_checkpoint=sam_checkpoint)

    # create a json file to store the masks
    masks_all_samples = []

    for idx, sample in enumerate(sample_list):
        img_file = sample + ".jpg"

        if idx % 10 == 0:
            print("Processing sample {} / {}".format(idx, len(sample_list)))
        
        image_path = os.path.join(input_path, img_file)
        assert os.path.exists(image_path), "Image file {} does not exist".format(image_path)

        image = cv2.imread(os.path.join(input_path, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = preprocess_images(image)

        masks = mask_generator.generate(image)

        sample_dict = {}

        sample_dict["image"] = img_file
        sample_dict["target_size"] = [image.shape[0], image.shape[1]]

        # convert masks to coco format
        masks_coco = []
        for mask in masks:
            binary_mask = mask['segmentation']
            mask_rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
            mask['segmentation'] = mask_rle
            masks_coco.append(mask)

        sample_dict["masks"] = masks_coco

        masks_all_samples.append(sample_dict)

        # # debug
        # if idx == 10:
        #     break

    # for large dataset, it is more reasonable to save it as the h5 file
    # save masks as the h5 file
    h5_save_path = os.path.join(output_path, "masks.h5")

    # convert dict to string
    # https://stackoverflow.com/questions/16494669/how-to-store-dictionary-in-hdf5-dataset
    masks_all_samples_str = []
    for sample in masks_all_samples:
        masks_all_samples_str.append(str(sample))

    with h5py.File(h5_save_path, 'w') as f:
        f.create_dataset('masks', data=masks_all_samples_str)
    


def main():
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    sam_checkpoint = args.sam_checkpoint

    process_dataset(input_path=input_path, output_path=output_path, sam_checkpoint=sam_checkpoint)
    


if __name__ == "__main__":
    main()