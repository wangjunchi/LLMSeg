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

class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "bf16",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        reasonseg_train_sam_mask_helper=None,
        reasonseg_val_sam_mask_helper=None,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
            
            if split == "train":
                assert reasonseg_train_sam_mask_helper is not None
                self.reasonseg_train_sam_mask_helper = reasonseg_train_sam_mask_helper
            elif split == "val":
                assert reasonseg_val_sam_mask_helper is not None
                self.reasonseg_val_sam_mask_helper = reasonseg_val_sam_mask_helper
        

        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))


        # # breakpoint
        # import pdb; pdb.set_trace()

        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    "reason_seg",
                    reason_seg_data,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        return self.samples_per_epoch

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
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        # print("image_path: ", image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        # get sam segs
        image_name = os.path.basename(image_path)
        split = os.path.basename(os.path.dirname(image_path))
        if split == "train":
            segs_dict = self.reasonseg_train_sam_mask_helper.extract_sam_segs(image_name)
        elif split == "val":
            segs_dict = self.reasonseg_val_sam_mask_helper.extract_sam_segs(image_name)

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

            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
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
