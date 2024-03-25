import glob
import os
import random
import copy
import json 

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .vqa_dataset import VQADataset
from .sam_mask_reader import SAM_Mask_Reader
from .utils import compute_all_iou


def collate_fn_new(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    sam_segs_list = []
    ious_list = []
    iops_list = []
    inferences = []
    origin_segs_list = []
    bbox_list = []
    
    for data in batch:
        image_path_list.append(data.get('image_path'))
        images_list.append(data.get('images'))
        images_clip_list.append(data.get('images_clip'))
        conversation_list.extend(data.get('conversations', []))
        label_list.append(data.get('label'))
        masks_list.append(data.get('masks').float())
        resize_list.append(data.get('resize'))
        questions_list.append(data.get('questions'))
        sampled_classes_list.append(data.get('sampled_classes'))
        cnt += len(data.get('conversations', []))
        offset_list.append(cnt)
        sam_segs_list.append(data.get('segs'))
        ious_list.append(data.get('ious'))
        iops_list.append(data.get('iops'))
        inferences.append(data.get('inference'))
        origin_segs_list.append(data.get('segs_origin', None))  # optional
        bbox_list.append(data.get('bbox', None))  # optional

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break


            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "sam_segs_list": sam_segs_list,
        "sam_ious_list": ious_list,
        "sam_iops_list": iops_list,
        "origin_segs_list": origin_segs_list,
        "bbox_list": bbox_list,
    }


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    sam_segs_list = []
    ious_list = []
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        segs,
        ious,
        inference,

    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        sam_segs_list.append(segs)
        ious_list.append(ious)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break


            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "sam_segs_list": sam_segs_list,
        "sam_ious_list": ious_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        sam_masks_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        coco2017_sam_mask_helper=None,
        coco2014_sam_mask_helper=None,
        voc2010_sam_mask_helper=None,
        ade20k_sam_mask_helper=None,
        mapillary_sam_mask_helper=None,
        saiapr_sam_mask_helper=None,
        reasonseg_train_sam_mask_helper=None,
        reasonseg_val_sam_mask_helper=None,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.coco2017_sam_mask_helper = coco2017_sam_mask_helper
        self.coco2014_sam_mask_helper = coco2014_sam_mask_helper
        self.voc2010_sam_mask_helper = voc2010_sam_mask_helper
        self.ade20k_sam_mask_helper = ade20k_sam_mask_helper
        self.mapillary_sam_mask_helper = mapillary_sam_mask_helper
        self.saiapr_sam_mask_helper = saiapr_sam_mask_helper
        self.reasonseg_train_sam_mask_helper = reasonseg_train_sam_mask_helper
        self.reasonseg_val_sam_mask_helper = reasonseg_val_sam_mask_helper

        self.sem_datasets = sem_seg_data.split("||")
        self.sam_masks_dir = sam_masks_dir

        if "refer_seg" in self.datasets:
            self.coco2014_sam_mask_helper = SAM_Mask_Reader(
                os.path.join(sam_masks_dir, "coco2014", "masks.json")
            )
            self.saiapr_sam_mask_helper = SAM_Mask_Reader(
                os.path.join(sam_masks_dir, "saiapr", "masks.json")
            )
        if "reason_seg" in self.datasets:
            self.reasonseg_train_sam_mask_helper = SAM_Mask_Reader(
                os.path.join(sam_masks_dir, "reason_seg", "ReasonSeg", "train", "masks.json")
            )
        if "vqa" in self.datasets:
            if self.coco2017_sam_mask_helper is None:
                self.coco2017_sam_mask_helper = SAM_Mask_Reader(
                    os.path.join(sam_masks_dir, "coco2017", "masks.json")
                )
        if "sem_seg" in self.datasets:
            if "ade20k" in self.sem_datasets:
                self.ade20k_sam_mask_helper = SAM_Mask_Reader(
                    os.path.join(sam_masks_dir, "ade20k", "masks.json")
                )
            if "mapillary" in self.sem_datasets:
                self.mapillary_sam_mask_helper = SAM_Mask_Reader(
                    os.path.join(sam_masks_dir, "mapillary", "masks.json")
                )
            if "cocostuff" in self.sem_datasets or "paco_lvis" in self.sem_datasets:
                if self.coco2017_sam_mask_helper is None:
                    self.coco2017_sam_mask_helper = SAM_Mask_Reader(
                        os.path.join(sam_masks_dir, "coco2017", "masks.json")
                    )
            if "pascal_part" in self.sem_datasets:
                self.voc2010_sam_mask_helper = SAM_Mask_Reader(
                    os.path.join(sam_masks_dir, "voc2010", "masks.json")
                )
        
        

        # # for debug
        # sam_mask_dir = "/home/leikel/junchi/processed_data"
        # self.coco2014_sam_mask_helper = SAM_Mask_Reader(
        #                      os.path.join(sam_mask_dir, "coco2014", "masks.json")) 
        # self.saiapr_sam_mask_helper = SAM_Mask_Reader(
        #                     os.path.join(sam_mask_dir, "saiapr", "masks.json"))
        # self.reasonseg_train_sam_mask_helper = SAM_Mask_Reader(
        #                     os.path.join(sam_mask_dir, "reason_seg", "ReasonSeg", "train", "masks.json"))
        # self.reasonseg_val_sam_mask_helper = SAM_Mask_Reader(
        #                     os.path.join(sam_mask_dir, "reason_seg", "ReasonSeg", "val", "masks.json"))
    


        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        coco2017_sam_mask_helper=self.coco2017_sam_mask_helper,
                        mapillary_sam_mask_helper=self.mapillary_sam_mask_helper,
                        ade20k_sam_mask_helper=self.ade20k_sam_mask_helper,
                        voc2010_sam_mask_helper=self.voc2010_sam_mask_helper,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        coco2014_sam_mask_helper=self.coco2014_sam_mask_helper,
                        saiapr_sam_mask_helper=self.saiapr_sam_mask_helper,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        coco2017_sam_mask_helper=self.coco2017_sam_mask_helper,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        reasonseg_train_sam_mask_helper=self.reasonseg_train_sam_mask_helper,
                        reasonseg_val_sam_mask_helper=self.reasonseg_val_sam_mask_helper,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        return data[0]
        # inference = False
        # datasample = data[0]
        # datasample["inference"] = False
        # return *data[0], inference


class ValDataSet_ReasonSeg(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255
    
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        image_size=1024,
        reasonseg_val_sam_mask_helper=None,
    ):
        self.base_image_dir = base_image_dir
        val_dataset = "ReasonSeg|val"
        splits = val_dataset.split("|")
        # import pdb; pdb.set_trace()
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        else:
            print("currently only support reason_seg val set")
            raise ValueError("not supported")
        
        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.reasonseg_val_sam_mask_helper = reasonseg_val_sam_mask_helper
        assert self.reasonseg_val_sam_mask_helper is not None

    def __len__(self):
        return len(self.images)
    
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
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_path = image_path.replace(".jpg", ".json")
        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        # for validation set, we only use the first sentence
        sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks_list = [mask_json]

        masks_np = np.stack(masks_list, axis=0)
        masks = torch.from_numpy(masks_np)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        # get SAM segs
        image_name = os.path.basename(image_path)
        segs_dict = self.reasonseg_val_sam_mask_helper.extract_sam_segs(image_name)

        segs_square = segs_dict["segs_square"]
        segs_origin = segs_dict["segs_origin"]
        bbox = segs_dict["bbox"]

        segs_square = torch.from_numpy(segs_square).permute(2, 0, 1).contiguous() # (K, H, W)
        segs = F.interpolate(segs_square.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False, antialias=True) # (1, K, 64, 64)
        segs = segs.squeeze(0)

        # gt iou is not needed for validation
        inference = True
        # return (
        #     image_path,
        #     image,
        #     image_clip,
        #     conversations,
        #     masks,
        #     labels,
        #     resize,
        #     None,   # questions
        #     None,   # sampled_sents
        #     segs,
        #     None, # gt-ious
        #     inference
        # )

        return {
            'image_path': image_path,
            'images': image,
            'images_clip': image_clip,
            'conversations': conversations,
            'masks': masks,
            'label': labels,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,   # Assuming 'sampled_sents' is equivalent to 'sampled_classes' from your previous code
            'segs': segs,
            'ious': None,  # Assuming 'gt-ious' is equivalent to 'ious' from your previous code
            'inference': inference,
            'segs_origin': segs_origin,
            'bbox': bbox,
        }


class ValDataSet_LLMSeg(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255
    
    def __init__(
        self,
        json_path,
        tokenizer,
        vision_tower,
        image_size=896,
        egoobjects_sam_mask_helper=None,
        coco_sam_mask_helper=None,
        coco_image_dir=None,
        ego_objects_image_dir=None,
    ):
        self.json_path = json_path
        
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.egoobjects_sam_mask_helper = egoobjects_sam_mask_helper
        self.coco_sam_mask_helper = coco_sam_mask_helper

        self.coco_image_dir = coco_image_dir
        self.ego_objects_image_dir = ego_objects_image_dir
        assert self.coco_sam_mask_helper is not None
        assert self.coco_image_dir is not None
        
        self.json_data = json.load(open(json_path, "r"))
        self.samples = self.load_all_samples()

        # shuffle samples with fixed seed
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return 100
        # return len(self.samples)
    
    def load_all_samples(self):
        samples = []
        with open(self.json_path, "r") as f:
            data = self.json_data
            images = data.keys()
            for image in images:
                sample = data[image]
                from_dataset = sample['from_dataset']
                if sample['from_dataset'] == "ego_objects":
                    # skip ego_objects temporarily
                    image_path = os.path.join(self.ego_objects_image_dir, image) 
                else:
                    image_path = os.path.join(self.coco_image_dir, image)
                qas = sample['qa_pairs']
                for qa in qas:
                    question = qa['question']
                    answer = qa['answer']
                    segmentation_path = qa['segmentation_path']

                    samples.append({
                        'image_path': image_path,
                        'question': question,
                        'answer': answer,
                        'from_dataset': from_dataset, # 'coco' or 'ego_objects'
                        'segmentation_path': segmentation_path,
                    })

        return samples

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
        # image_path = self.images[idx]
        sample = self.samples[idx]
        image_path = sample['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sampled_sents = [sample['question']]
        is_sentence = True

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        segmentation_path = sample['segmentation_path']
        masks_np = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        # convert to binary mask
        masks_np = (masks_np > 0).astype(np.float32)
        
        # import pdb; pdb.set_trace()

        masks = torch.from_numpy(masks_np).unsqueeze(0)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

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
        bbox = segs_dict["bbox"]

        segs_square = torch.from_numpy(segs_square).permute(2, 0, 1).contiguous() # (K, H, W)
        segs = F.interpolate(segs_square.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False, antialias=True) # (1, K, 64, 64)
        segs = segs.squeeze(0)

        # gt iou is not needed for validation
        inference = True

        return {
            'image_path': image_path,
            'images': image,
            'images_clip': image_clip,
            'conversations': conversations,
            'masks': masks,
            'label': labels,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,   # Assuming 'sampled_sents' is equivalent to 'sampled_classes' from your previous code
            'segs': segs,
            'ious': None,  # Assuming 'gt-ious' is equivalent to 'ious' from your previous code
            'inference': inference,
            'segs_origin': segs_origin,
            'bbox': bbox,
        }




        


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 896
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        reasonseg_val_sam_mask_helper=None,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
