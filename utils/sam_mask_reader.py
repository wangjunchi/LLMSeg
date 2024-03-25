import os
import json
from typing import List, Dict

import numpy as np
import torch
import pycocotools.mask as mask_util
from skimage.transform import resize
import cv2
import time
class SAM_Mask_Reader:

    def __init__(self, json_dir) -> None:
        self.json_dir = json_dir

        print("reading sam mask json: ", json_dir)
        start_time = time.time()
        self.mask_list = self.read_mask_json(json_dir)
        end_time = time.time()
        print(f"read sam mask json takes {end_time - start_time} seconds")

        print("building sam mask index")
        start_time = time.time()
        self.sam_mask_index = self.build_sam_mask_index()
        end_time = time.time()
        print(f"build sam mask index takes {end_time - start_time} seconds")

        print("sam_mask_list: ", len(self.mask_list))
        print("sam_mask_index: ", len(self.sam_mask_index))

    def read_mask_json(self, path: str):
        with open(path, "r") as f:
            mask_list = json.load(f)
        return mask_list
    
    def build_sam_mask_index(self): 
        sam_mask_index = {}
        for i, sample in enumerate(self.mask_list):
            sample_name = sample["image"]
            sam_mask_index[sample_name] = i

        return sam_mask_index
    
    def get_sam_mask_index(self, image_name: str):
        if image_name not in self.sam_mask_index:
            raise ValueError(f"image_name: {image_name} not in sam_mask_index")
        return self.sam_mask_index[image_name]
    
    def preprocess_mask(self, masks: np.ndarray):
        # masks: (H, W, K)

        # convert mask to float
        masks = masks.astype(np.float64)
        # padding to square
        h, w, _ = masks.shape
        padh = max(h, w) - h
        padw = max(h, w) - w
        masks = np.pad(masks, ((0, padh), (0, padw), (0, 0)), mode="constant", constant_values=0)

        assert masks.shape[0] == masks.shape[1]
        assert masks.shape[0] == max(h, w)

        # # resize to 64x64
        # mask = resize(mask, (64, 64), anti_aliasing=True)

        return masks


    def extract_sam_segs(self, image_name: str):
        
        index = self.get_sam_mask_index(image_name)
        sam_masks = self.mask_list[index]

        seg_list = []
        seg_np_list_large = []
        # extract binary seg

        # sort sam_masks by area
        masks = sam_masks['masks']
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

        if len(masks_sorted) > 50:
            masks_sorted = masks_sorted[:50]
            # print("Warning: too many sam masks, only use the top 50, image_name: ", image_name)

        rle_segs = [mask['segmentation'] for mask in masks_sorted]
        segs_origin = mask_util.decode(rle_segs)  # (H, W, K)

        segs_square = self.preprocess_mask(segs_origin)

        bbox = [mask['bbox'] for mask in masks_sorted]

        # for mask in sam_masks['masks']:
        #     seg_rle = mask['segmentation']
        #     # decode coco rle format
        #     seg = mask_util.decode(seg_rle)
        #     seg_np_list_large.append(seg)
        #     # padding to square and resize to 64x64
        #     seg_small = self.preprocess_mask(seg)
        #     # to tensor
        #     seg_small = torch.from_numpy(seg_small).float()

        #     seg_list.append(seg_small.unsqueeze(0))
        
        # # concat seg_list
        # segs_tensor_small = torch.cat(seg_list, dim=0)
        # return segs_square, segs

        return {
            "segs_square": segs_square,
            "segs_origin": segs_origin,
            "bbox": bbox
        }
    
    


    