from enum import Enum

import numpy as np
import torch
import torch.distributed as dist
import cv2
from typing import List
from skimage.transform import resize

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explaination.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict, torch_dtype=torch.bfloat16):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
            if k == "images" or k == "images_clip":
                input_dict[k] = input_dict[k].to(dtype=torch_dtype)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
            if k == "sam_segs_list":
                input_dict[k] = [ele.to(dtype=torch_dtype) for ele in input_dict[k]] 
    return input_dict


def compute_iop(seg: np.ndarray, gt: np.ndarray):
    # seg and gt are both binary masks
    assert seg.shape == gt.shape

    if seg.max() > 1 or gt.max() > 1:
        raise ValueError("seg and gt should be binary masks")
    
    # comput iou with all connected components in gt
    intersection = np.logical_and(seg, gt)
    union = np.logical_or(seg, gt)
    # iou = np.sum(intersection) / np.sum(union)
    iop = np.sum(intersection) / np.sum(seg)

    max_iop = iop

    return max_iop


def compute_iou(seg: np.ndarray, gt: np.ndarray):
    # seg and gt are both binary masks
    assert seg.shape == gt.shape

    if seg.max() > 1 or gt.max() > 1:
        raise ValueError("seg and gt should be binary masks")
    
    # comput iou with all connected components in gt
    intersection = np.logical_and(seg, gt)
    union = np.logical_or(seg, gt)
    iou = np.sum(intersection) / np.sum(union)

    max_iou = iou

    # # Find connected components in the ground truth mask
    # num_labels, labels = cv2.connectedComponents(gt)


    # # Compute IoU for each component
    # for i in range(1, num_labels):  # Start from 1 to ignore background
    #     component_mask = (labels == i).astype(np.uint8)
    #     # compute the iou between seg and component_mask
    #     intersection = np.logical_and(seg, component_mask)
    #     union = np.logical_or(seg, component_mask)
    #     iou = np.sum(intersection) / np.sum(union)

    #     max_iou = max(max_iou, iou)

    return max_iou

# def compute_iou(seg: np.ndarray, gt: np.ndarray):
#     # seg and gt are both binary masks
#     assert seg.shape == gt.shape
#     if seg.max() > 1 or gt.max() > 1:
#         raise ValueError("seg and gt should be binary masks")
    
#     intersection = np.logical_and(seg, gt)
#     union = np.logical_or(seg, gt)
#     iou = np.sum(intersection) / np.sum(union)

#     return iou
    
def compute_all_iou(segs: List[np.ndarray], gt: np.ndarray):
    # conptue the iou between segs and gt
    # segs: list of (H, W) : may be resized to 1024 if the original size is too large
    # gt: (H', W') original size of the image
    H, W, K = segs.shape

    gt = resize(gt, (H, W), anti_aliasing=False, preserve_range=True, order=0)
    
    ious = []
    for i in range(K):
        seg_i = segs[:, :, i]

        assert seg_i.shape == gt.shape

        iou = compute_iou(seg_i, gt)

        ious.append(iou)

    return np.array(ious)


def compute_all_iop(segs: List[np.ndarray], gt: np.ndarray):
    # conptue the iou between segs and gt
    # segs: list of (H, W) : may be resized to 1024 if the original size is too large
    # gt: (H', W') original size of the image
    H, W, K = segs.shape

    gt = resize(gt, (H, W), anti_aliasing=False, preserve_range=True, order=0)
    
    iops = []
    for i in range(K):
        seg_i = segs[:, :, i]

        assert seg_i.shape == gt.shape

        iop = compute_iop(seg_i, gt)

        iops.append(iop)

    return np.array(iops)