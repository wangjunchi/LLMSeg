import numpy as np
import os 
import cv2 
from typing import List
import json
import h5py
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

dataset_root = "/home/leikel/junchi/lisa_dataset/ego_objects"
output_root = "/home/leikel/junchi/lisa_dataset/ego_objects"
sam_checkpoint = "/home/leikel/junchi/pretrained_weights/SAM/sam_vit_h_4b8939.pth"


# test set not available for ReasonSeg
available_dataset_type = ["train", "validation", "test"]

def get_all_samples() -> List[str]:
    
    json_path = "/home/leikel/junchi/ReasonCOCO/post_processing/split"

    all_samples = []

    for dataset_type in available_dataset_type:
        with open(os.path.join(json_path, dataset_type + ".json"), "r") as f:
            data = json.load(f)
            images = data.keys()
            for image in images:
                sample = data[image]
                if sample['from_dataset'] == "ego_objects":
                    all_samples.append(image)

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


def process_dataset() -> None:

    
    input_path = os.path.join(dataset_root, "images")
    output_path = output_root

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    sample_list = get_all_samples()

    print("we have {} samples".format(len(sample_list)))

    mask_generator = init_SAM_everything(model_type="vit_h", sam_checkpoint=sam_checkpoint)

    # create a json file to store the masks
    masks_all_samples = []

    for sample in tqdm(sample_list[:]):
        img_file = sample 

        # print("Processing sample {} / {}".format(idx, len(sample_list)))
        
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
            mask_rle['counts'] = mask_rle['counts'].decode()
            mask['segmentation'] = mask_rle
            masks_coco.append(mask)

        sample_dict["masks"] = masks_coco

        masks_all_samples.append(sample_dict)

    # save to json
    json_save_path = os.path.join(output_path, "masks.json")

    with open(json_save_path, 'w') as f:
        json.dump(masks_all_samples, f)

        
def main():
    process_dataset()


if __name__ == "__main__":
    main()