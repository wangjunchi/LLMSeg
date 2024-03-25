import argparse
import os 
import shutil
import sys
import time
from functools import partial
import copy

import deepspeed 
import numpy as np
import torch 
import tqdm 
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import wandb
import cv2


from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, collate_fn, ValDataSet_ReasonSeg, collate_fn_new, ValDataSet_LLMSeg
from utils.sam_mask_reader import SAM_Mask_Reader
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

# set random seed
# import random
# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)


# os.environ["NCCL_P2P_DISABLE"] = "1"

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="/cluster/work/cvl/leikel/junchi/LISA/pretrained_weights/LLaVA-lightning-7B-v1/"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=896, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="refer_seg||reason_seg", type=str
    )
    
    parser.add_argument("--sample_rates", default="10, 1", type=str)

    # parser.add_argument(
    #     "--dataset", default="refer_seg", type=str
    # )
    # parser.add_argument("--sample_rates", default="1", type=str)

    # parser.add_argument(
    #     "--dataset", default="reason_seg", type=str
    # )
    # parser.add_argument("--sample_rates", default="1", type=str)

    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="/cluster/scratch/leikel/junchi/lisa_dataset", type=str)
    parser.add_argument("--sam_masks_dir", default="/home/leikel/junchi/processed_data", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="debug", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--align_loss_weight", default=1.0, type=float)
    parser.add_argument("--regression_loss_weight", default=1.0, type=float)
    # parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    # parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/cluster/work/cvl/leikel/junchi/LISA/pretrained_weights/SAM/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--iou_selection_only", action="store_true", default=False)
    return parser.parse_args(args)


def init_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    _ = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    return tokenizer

def init_LISA_model(args, tokenizer):
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "align_loss_weight": args.align_loss_weight,
        "regression_loss_weight": args.regression_loss_weight,
        # "dice_loss_weight": args.dice_loss_weight,
        # "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()  # CLIP
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_lisa_modules(model.get_model().config)  # SAM and others

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    # init LoRA
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "lisa_attention_layers",
                                "lisa_final_attn",
                                "lisa_norm_final_attn",
                                "lisa_iou_head",
                                "lisa_embedding_head",
                                "lisa_dino_conv",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_hidden_fcs, lm_head, embed_tokens trainable
    # two fcs belong to LISA
    # lm_head and embed_tokens belong to llava
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "text_hidden_fcs",
                          "lisa_attention_layers", "lisa_final_attn", "lisa_norm_final_attn",
                          "lisa_iou_head", "lisa_embedding_head", "lisa_dino_conv"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    model.print_trainable_parameters()
    return model

def init_training_dataset(args, tokenizer):

    world_size = torch.cuda.device_count()
    train_dataset = HybridDataset(
        args.dataset_dir,
        args.sam_masks_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
    )

    return train_dataset


def init_validation_dataset(args, tokenizer):
    if args.no_eval:
        return None

    sam_mask_dir = "/home/leikel/junchi/processed_data/"
    json_path ="/home/leikel/junchi/ReasonCOCO/post_processing/split/validation.json"
    coco_image_dir='/home/leikel/junchi/lisa_dataset/coco/train2017'
    ego_objects_image_dir = '/home/leikel/junchi/lisa_dataset/ego_objects/images'

    coco2017_sam_mask_helper = SAM_Mask_Reader(
                            os.path.join(sam_mask_dir, "coco2017", "masks.json")) 
    egoobjects_sam_mask_helper = SAM_Mask_Reader(
                            os.path.join(sam_mask_dir, "ego_objects", "masks.json"))
    val_dataset = ValDataSet_LLMSeg(
        json_path,
        tokenizer,
        args.vision_tower,
        args.image_size,
        egoobjects_sam_mask_helper=egoobjects_sam_mask_helper,
        coco_sam_mask_helper=coco2017_sam_mask_helper,
        coco_image_dir=coco_image_dir,
        ego_objects_image_dir=ego_objects_image_dir,
    )

    return val_dataset


def init_deepseed_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            "ignore_unused_parameters": True,
        },
    }

    return ds_config



def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)

        # init wandb
        exp_name = args.exp_name + time.strftime("_%Y_%m_%d_%H_%M_%S")
        wandb.init(
            project="New-Thesis",
            name=exp_name,
            config=args,
        )
    else:
        writer = None

    # set random seed
    import random
    random.seed(0+args.local_rank)
    np.random.seed(0+args.local_rank)

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    tokenizer = init_tokenizer(args)
    model = init_LISA_model(args, tokenizer)

    train_dataset = init_training_dataset(args, tokenizer)
    val_dataset = init_validation_dataset(args, tokenizer)
    # val_dataset = None
    if train_dataset is not None:
        print(f"Training with {len(train_dataset)} examples.")
    if val_dataset is not None:
        print(f"Validation with {len(val_dataset)} examples.")
    # init deepspeed distributed training
    ds_config = init_deepseed_config(args)
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn_new,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn_new,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
    )

    train_iter = iter(train_loader)

    if args.eval_only:
        # giou, ciou = validate_threshold_from_topIoU(val_loader, model_engine, 0, writer, args, threshold=0.5)
        # giou, ciou = validate_iou_iop(val_loader, model_engine, 0, writer, args, threshold=0.5)
        # giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        giou, ciou = validate_threshold(val_loader, model_engine, 0, writer, args, threshold=0.5)
        # for i in range(10):
        #     threshold = 0.1 * (i + 1)
        #     giou, ciou = validate_threshold(val_loader, model_engine, 0, writer, args, threshold=threshold)
        #     print("results from threshold {}: giou={}, ciou={}".format(threshold, giou, ciou))
        exit()

    best_score, cur_ciou = 0.0, 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            if not args.iou_selection_only:
                giou, ciou = validate_threshold(val_loader, model_engine, epoch, writer, args)
            print("results from threshold: giou={}, ciou={}".format(giou, ciou))

            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

            
        # save checkpoint
        if args.no_eval or is_best:
            # pass
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    align_losses = AverageMeter("AlignLoss", ":.4f")
    regression_losses = AverageMeter("RegressionLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            align_losses,
            regression_losses
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                print("error fetching data, skip it")
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            
            input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

            # print("forward for rank: ", args.local_rank)
            output_dict = model(**input_dict)
            # print("forward done for rank: ", args.local_rank)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            align_loss = output_dict["align_loss"]
            regression_loss = output_dict["regression_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            align_losses.update(align_loss.item(), input_dict["images"].size(0))
            regression_losses.update(regression_loss.item(), input_dict["images"].size(0))
            
            # print("backward for rank: ", args.local_rank)
            model.backward(loss)
            model.step()
            # print("backward done for rank: ", args.local_rank)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                align_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/align_loss", align_losses.avg, global_step
                )
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

                # log using wandb
                if args.local_rank == 0:
                    wandb.log(
                        {
                            "train/loss": losses.avg,
                            "train/ce_loss": ce_losses.avg,
                            "train/align_loss": align_losses.avg,
                            "train/regression_loss": regression_losses.avg,
                        },
                        step=epoch * args.steps_per_epoch + global_step,
                    )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            align_losses.reset()
            regression_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter
        

def validate(val_loader, model_engine, epoch, writer, args):
    print("start validating ###############################")
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

        with torch.no_grad():
            output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_similarity"][0]
        # get the seg with highest similarity
        max_idx = torch.argmax(pred_similarity).item()

        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        pred_seg = sam_segs[:, :, max_idx] # (H, W)
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        # send pred_seg and gt_mask to GPU
        pred_seg = pred_seg.cuda()
        gt_mask = gt_mask.cuda()

        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)

        assert pred_seg.shape == gt_mask.shape

        # compute IoU
        # Be careful, wrong result for uint8
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )

        acc_iou = intersection / (union + 1e-8)

        acc_iou[union == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

            # # compute iou
            # intersection, union, acc_iou = 0.0, 0.0, 0.0
            # for mask_i, output_i in zip(masks_list, output_list):
            #     intersection_i, union_i, _ = intersectionAndUnionGPU(
            #         output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            #     )
            #     intersection += intersection_i
            #     union += union_i
            #     acc_iou += intersection_i / (union_i + 1e-5)
            #     acc_iou[union_i == 0] += 1.0  # no-object target
            # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            # acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            # intersection_meter.update(intersection), union_meter.update(
            #     union
            # ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/giou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

        # log using wandb
        wandb.log(
            {
                "val/giou": giou,
                "val/ciou": ciou,
            },
            step=(epoch+1) * args.steps_per_epoch - 1,
        )
        
    return giou, ciou


def validate_threshold(val_loader, model_engine, epoch, writer, args, threshold=0.5):
    print("start validating ###############################")
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

        with torch.no_grad():
            output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_iou"][0]
        # print(pred_similarity)
        # get the seg with highest similarity
        max_ids = []
        for i in range(pred_similarity.shape[1]):
            if pred_similarity[0][i] > threshold:
                max_ids.append(i)


        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        # pred_seg = sam_segs[:, :, max_idx] # (H, W)
        # pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        pred_seg = np.zeros_like(sam_segs[:, :, 0])
        for i in max_ids:
            pred_seg += sam_segs[:, :, i]
        pred_seg = pred_seg > 0
        pred_seg = pred_seg.astype(np.uint8)

        # send pred_seg and gt_mask to GPU
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)

        # # resize pred_seg and gt_mask to 1024x1024
        # pred_seg = torch.nn.functional.interpolate(
        #     pred_seg.unsqueeze(0), size=(1024, 1024), mode="nearest"
        # ).squeeze(0)
        # gt_mask = torch.nn.functional.interpolate(
        #     gt_mask.unsqueeze(0), size=(1024, 1024), mode="nearest"
        # ).squeeze(0)

        pred_seg = pred_seg.cuda()
        gt_mask = gt_mask.cuda()

        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)

        assert pred_seg.shape == gt_mask.shape

        # compute IoU
        # Be careful, wrong result for uint8
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )

        acc_iou = intersection / (union + 1e-8)

        acc_iou[union == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

        if args.eval_only and args.visualize:
            # save the evaluation result
            image_path = input_dict['image_paths'][0]
            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))
                continue
            
            image_name = os.path.basename(image_path)
            # image and mask has the same shape
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pred_mask = pred_seg.detach().cpu().numpy()
            pred_mask = pred_mask[0]      
            pred_mask = pred_mask > 0      

            gt_mask = gt_mask.detach().cpu().numpy()
            gt_mask = gt_mask[0]
            gt_mask[gt_mask == 255] = 0  # ignored label
            gt_mask = gt_mask > 0

            overlap_image = image.copy()
            overlap_image[pred_mask] = (
                image * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            overlap_image = cv2.cvtColor(overlap_image, cv2.COLOR_RGB2BGR)
            
            overlap_image_gt = image.copy()
            overlap_image_gt[gt_mask] = (
                image * 0.5
                + gt_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[gt_mask]
            overlap_image_gt = cv2.cvtColor(overlap_image_gt, cv2.COLOR_RGB2BGR)

            # process pred_mask for save
            pred_mask = pred_mask.astype(np.uint8) * 255
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

            iou = acc_iou[1]

            # conversations is a string
            conversations = input_dict["conversation_list"][0]
            conversations = conversations.replace("<im_patch>", "")


            # save to dir
            save_dir = os.path.join(args.log_dir, "eval_vis_iop")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            all_iops = pred_similarity
            # save conversations as text
            with open(os.path.join(save_dir, image_name+".txt"), 'w') as file:
                file.write(conversations)
                file.write("\n\n")
                file.write("iou = "+str(iou))
                file.write("\n\n")
                file.write("all_iops = "+str(all_iops))

            # convert image back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # save images
            cv2.imwrite(os.path.join(save_dir, image_name+".png"), image)
            cv2.imwrite(os.path.join(save_dir, image_name+"_mask.png"), pred_mask)
            cv2.imwrite(os.path.join(save_dir, image_name+"_pred.png"), overlap_image)
            cv2.imwrite(os.path.join(save_dir, image_name+"_gt.png"), overlap_image_gt)

            # # compute iou
            # intersection, union, acc_iou = 0.0, 0.0, 0.0
            # for mask_i, output_i in zip(masks_list, output_list):
            #     intersection_i, union_i, _ = intersectionAndUnionGPU(
            #         output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            #     )
            #     intersection += intersection_i
            #     union += union_i
            #     acc_iou += intersection_i / (union_i + 1e-5)
            #     acc_iou[union_i == 0] += 1.0  # no-object target
            # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            # acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            # intersection_meter.update(intersection), union_meter.update(
            #     union
            # ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/giou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

        # log using wandb
        wandb.log(
            {
                "val/giou": giou,
                "val/ciou": ciou,
            },
            step=(epoch+1) * args.steps_per_epoch - 1,
        )
        
    return giou, ciou

def validate_iou_iop(val_loader, model_engine, epoch, writer, args, threshold=0.5):
    print("start validating using iou+iop ###############################")
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

        with torch.no_grad():
            output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_similarity"][0]
        pred_iop = output_dict["pred_iou"][0]

        # get the seg with highest similarity
        max_idx = torch.argmax(pred_similarity).item()

        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        # pred_seg = sam_segs[:, :, max_idx] # (H, W)
        # pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)

        max_ids = [max_idx]
        for i in range(pred_iop.shape[1]):
            if pred_iop[0][i] > threshold and i != max_idx:
                max_ids.append(i)

        pred_seg = np.zeros_like(sam_segs[:, :, 0])
        for i in max_ids:
            pred_seg += sam_segs[:, :, i]
        pred_seg = pred_seg > 0
        pred_seg = pred_seg.astype(np.uint8)

        # send pred_seg and gt_mask to GPU
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        pred_seg = pred_seg.cuda()
        gt_mask = gt_mask.cuda()

        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)

        assert pred_seg.shape == gt_mask.shape

        # compute IoU
        # Be careful, wrong result for uint8
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )

        acc_iou = intersection / (union + 1e-8)

        acc_iou[union == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/giou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

        # log using wandb
        wandb.log(
            {
                "val/giou": giou,
                "val/ciou": ciou,
            },
            step=(epoch+1) * args.steps_per_epoch - 1,
        )
        
    return giou, ciou

def validate_threshold_from_topIoU(val_loader, model_engine, epoch, writer, args, threshold=0.5):
    print("start validating using threshold from top IoU ###############################")
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict, torch_dtype=torch_dtype)

        with torch.no_grad():
            output_dict = model_engine(**input_dict)
        
        pred_similarity = output_dict["pred_similarity"][0]
        pred_iop = output_dict["pred_iou"][0]

        # get the seg with highest similarity
        max_idx = torch.argmax(pred_similarity).item()

        sam_segs = input_dict["origin_segs_list"][0] # (H, W, K)
        gt_mask = output_dict["gt_masks"][0] # (1, H', W')

        # pred_seg = sam_segs[:, :, max_idx] # (H, W)
        # pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)

        # max_ids = [max_idx]
        # for i in range(pred_iop.shape[1]):
        #     if pred_iop[0][i] > threshold and i != max_idx:
        #         max_ids.append(i)

        K = 5
        if K > pred_similarity.shape[-1]:
            K = pred_similarity.shape[-1]
        
        # import pdb; pdb.set_trace()


        topK_ids = torch.topk(pred_similarity[0], K, dim=0).indices
        max_ids = []
        for i in topK_ids:
            if pred_iop[0][i] > threshold:
                max_ids.append(i)

        pred_seg = np.zeros_like(sam_segs[:, :, 0])
        for i in max_ids:
            pred_seg += sam_segs[:, :, i]
        pred_seg = pred_seg > 0
        pred_seg = pred_seg.astype(np.uint8)

        # send pred_seg and gt_mask to GPU
        pred_seg = torch.from_numpy(pred_seg).unsqueeze(0) # (1, H, W)
        pred_seg = pred_seg.cuda()
        gt_mask = gt_mask.cuda()

        # resize if shape is not equal
        if pred_seg.shape != gt_mask.shape:
            pred_seg = torch.nn.functional.interpolate(
                pred_seg.unsqueeze(0), size=gt_mask.shape[1:], mode="nearest"
            ).squeeze(0)

        assert pred_seg.shape == gt_mask.shape

        # compute IoU
        # Be careful, wrong result for uint8
        intersection, union, _ = intersectionAndUnionGPU(
            pred_seg.int().contiguous(), gt_mask.int().contiguous(), 2
        )

        acc_iou = intersection / (union + 1e-8)

        acc_iou[union == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/giou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

        # log using wandb
        wandb.log(
            {
                "val/giou": giou,
                "val/ciou": ciou,
            },
            step=(epoch+1) * args.steps_per_epoch - 1,
        )
        
    return giou, ciou

if __name__ == "__main__":
    main(sys.argv[1:])
