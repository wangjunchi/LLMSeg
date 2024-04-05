import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from transformers import CLIPImageProcessor
import pycocotools.mask as mask_util

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

from .sam_mask_reader import SAM_Mask_Reader
from .utils import compute_all_iou, compute_all_iop

class LLMSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255

    def __init__(
        self,
        json_path,
        tokenizer,
        vision_tower,
        precision: str = "bf16",
        image_size: int = 896,
        egoobjects_sam_mask_helper=None,
        coco_sam_mask_helper=None,
        coco_image_dir=None,
        ego_objects_image_dir=None,
    ):
        self.json_path = json_path
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.egoobjects_sam_mask_helper = egoobjects_sam_mask_helper
        self.coco_sam_mask_helper = coco_sam_mask_helper

        self.coco_image_dir = coco_image_dir
        self.ego_objects_image_dir = ego_objects_image_dir
        assert self.coco_sam_mask_helper is not None
        assert self.egoobjects_sam_mask_helper is not None

        self.json_data = json.load(open(json_path, "r"))
        self.samples = self.load_all_samples()

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

    def load_all_samples(self):
        samples = []
        with open(self.json_path, "r") as f:
            data = self.json_data
            images = data.keys()
            for image in images:
                sample = data[image]
                from_dataset = sample['from_dataset']
                if sample['from_dataset'] == "ego_objects":
                    image_path = os.path.join(self.ego_objects_image_dir, image) 
                else:
                    image_path = os.path.join(self.coco_image_dir, image)
                qas = sample['qa_pairs']
                for qa in qas:
                    question = qa['question']
                    answer = qa['answer']
                    # segmentation_path = qa['segmentation_path']

                    # read rle mask from json
                    rle_seg = qa['rle_seg']

                    samples.append({
                        'image_path': image_path,
                        'question': question,
                        'answer': answer,
                        'from_dataset': from_dataset, # 'coco' or 'ego_objects'
                        'rle_seg': rle_seg,
                        # 'segmentation_path': segmentation_path,
                    })

        return samples
        
    def __len__(self):
        return len(self.samples)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sampled_sents = [sample['question']]
        is_sentence = True

        # print("image_path: ", image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # segmentation_path = sample['segmentation_path']
        # masks_np = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        # # convert to binary mask
        # masks_np = (masks_np > 0).astype(np.float32)

        rle_seg = sample['rle_seg']
        mask = mask_util.decode(rle_seg)
        masks_np = (mask > 0).astype(np.float32)
        
        # import pdb; pdb.set_trace()

        sampled_masks = [masks_np]

        # get SAM segs
        image_name = os.path.basename(image_path)
        # segs_dict = self.reasonseg_val_sam_mask_helper.extract_sam_segs(image_name)
        from_dataset = sample['from_dataset']
        if from_dataset == "coco":
            segs_dict = self.coco_sam_mask_helper.extract_sam_segs(image_name)
        elif from_dataset == "ego_objects":
            segs_dict = self.egoobjects_sam_mask_helper.extract_sam_segs(image_name)
        else:
            raise ValueError("from_dataset not supported")
        
        segs_square = segs_dict["segs_square"]
        segs_origin = segs_dict["segs_origin"]

        # import pdb; pdb.set_trace()
        # convert segs_square to tensor and resize to 64x64
        segs_square = torch.from_numpy(segs_square).permute(2, 0, 1).contiguous() # (K, H, W)
        # print("dtype: ", segs_square.dtype)
        # print("shape: ", segs_square.shape)
        segs = F.interpolate(segs_square.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False, antialias=True) # (1, K, 64, 64)
        segs = segs.squeeze(0) # (K, 64, 64)
        
        # compute iou
        sampled_ious = [
            compute_all_iou(segs_origin, mask.astype(np.uint8)) for mask in sampled_masks
        ]

        sampled_iops = [
            compute_all_iop(segs_origin, mask.astype(np.uint8)) for mask in sampled_masks
        ]

        precision_type = torch.float32
        if self.precision == "fp16":
            precision_type = torch.float16
        elif self.precision == "bf16":
            precision_type = torch.bfloat16
        
        segs = segs.to(precision_type)

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image_name = image_path.split("/")[-1]
        # if self.explanatory != -1 and image_name in self.img_to_explanation:
        #     if random.random() < self.explanatory:
        #         choice = 2
        #     else:
        #         choice = random.randint(0, 1)

        # simplify the problem: only segment the object, do not need any explanation
        choice = 0

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # # add explanation if applicable
            # img_name = image_path.split("/")[-1]
            # if self.explanatory != -1 and img_name in self.img_to_explanation:
            #     if choice == 0:  # [SEG] token
            #         answers.append(random.choice(self.answer_list))
            #     else:
            #         raise ValueError("Not implemented yet.")
            # else:
            answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        image_name = image_path.split("/")[-1]

        masks = np.stack(sampled_masks, axis=0)
        ious = np.stack(sampled_ious, axis=0)
        iops = np.stack(sampled_iops, axis=0)
        masks = torch.from_numpy(masks)
        ious = torch.from_numpy(ious)
        iops = torch.from_numpy(iops)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label


        return {
            'image_path': image_path,
            'images': image,
            'images_clip': image_clip,
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': questions,
            'sampled_classes': sampled_sents,  
            'segs': segs,
            'ious': ious, 
            'iops': iops,
            'segs_origin': None,
            'bbox': None,
            'inference': False,
        }
