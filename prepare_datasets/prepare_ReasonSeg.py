import numpy as np
import os 
import cv2 
from typing import List
import json
import h5py
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


dataset_root = "/cluster/scratch/leikel/junchi/lisa_dataset/reason_seg/ReasonSeg/"
output_root = "/cluster/home/leikel/junchi/processed_data/reason_seg/ReasonSeg/"
sam_checkpoint = "/cluster/home/leikel/junchi/segment-anything/checkpoints/sam_vit_h_4b8939.pth"


# test set not available for ReasonSeg
available_dataset_type = ["train", "val"]

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


def process_dataset(dataset_type: str = "train") -> None:
    assert dataset_type in available_dataset_type, "dataset_type must be one of {}".format(available_dataset_type)
    print("Processing dataset {}".format(dataset_type))
    
    input_path = os.path.join(dataset_root, dataset_type)
    output_path = os.path.join(output_root, dataset_type)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    sample_list = get_all_samples(os.path.join(dataset_root, dataset_type))

    mask_generator = init_SAM_everything(model_type="vit_h", sam_checkpoint=sam_checkpoint)

    # create a json file to store the masks
    masks_all_samples = []

    for idx, sample in enumerate(sample_list):
        img_file = sample + ".jpg"

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

    # for large dataset, it is more reasonable to save it as the h5 file
    # save masks as the h5 file
    h5_save_path = os.path.join(output_path, dataset_type + "_masks.h5")

    # convert dict to string
    # https://stackoverflow.com/questions/16494669/how-to-store-dictionary-in-hdf5-dataset
    masks_all_samples_str = []
    for sample in masks_all_samples:
        masks_all_samples_str.append(str(sample))

    with h5py.File(h5_save_path, 'w') as f:
        f.create_dataset('masks', data=masks_all_samples_str)
    

        
def main():
    for dataset_type in available_dataset_type:
        process_dataset(dataset_type=dataset_type)


if __name__ == "__main__":
    main()