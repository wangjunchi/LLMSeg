from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h

from .loss import sigmoid_align_loss, softmax_align_loss, iou_regression_loss
from .transformer import Attention, MLPBlock, LISA_TwoWayAttentionBlock

class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        print("Initializing LISA modules...")        
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False

        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # DINO-V2
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.visual_model_dinov2 = dinov2_vitl14
        for param in self.visual_model_dinov2.parameters():
            param.requires_grad = False

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        
        # # add fc for mask embedding
        # mask_fc = [
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(out_dim, out_dim),
        #     nn.Dropout(0.0),
        # ]
        # # redudant code, but for clarity
        # self.mask_hidden_fcs = nn.ModuleList([nn.Sequential(*mask_fc)])
        # self.mask_hidden_fcs.train()
        # for param in self.mask_hidden_fcs.parameters():
        #     param.requires_grad = True

        # # learnable temperature and bias
        # self.temperature = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        # self.bias = nn.Parameter(torch.tensor(-10.0), requires_grad=True)

        # # use attention to replace the fc layer
        # self.lisa_mask_self_attn= Attention(256, 8)
        # self.lisa_norm1 = nn.LayerNorm(256)
        # self.lisa_cross_attn_mask_to_text = Attention(256, 8)
        # self.lisa_norm2 = nn.LayerNorm(256)
        # self.lisa_mlp = MLPBlock(256, 2048, torch.nn.ReLU)

        # 1x1 conv to reduce the dimension of the image feature to 256
        self.lisa_dino_conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)

        self.lisa_attention_layers = nn.ModuleList()
        depth = 2
        for i in range(depth):
            self.lisa_attention_layers.append(
                LISA_TwoWayAttentionBlock(
                    embedding_dim=256,
                    num_heads=8,
                    mlp_dim=2048,
                    attention_downsample_rate=1,
                )
            )
        self.lisa_final_attn = Attention(
            embedding_dim=256, num_heads=8, downsample_rate=1
        )
        self.lisa_norm_final_attn = nn.LayerNorm(256)

        self.lisa_iou_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.lisa_embedding_head = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        )



class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
            self.align_loss_weight = kwargs.pop("align_loss_weight", 1.0)
            self.regression_loss_weight = kwargs.pop("regression_loss_weight", 1.0)
            #self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            #self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings
    
    def get_dinov2_visual_embs(self, pixel_values: torch.FloatTensor):

        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings_dict = self.model.visual_model_dinov2.forward_features(pixel_values[i].unsqueeze(0))
                image_embeddings = image_embeddings_dict['x_norm_patchtokens']
                # 1*4096*1024 -> 1*1024*256*256
                image_embeddings = image_embeddings.permute(0, 2, 1).reshape(1, 1024, 64, 64)
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def mask_pooling(self, image_embeddings: torch.FloatTensor, weight_maps: torch.FloatTensor):
        # image_embeddings: [256, 64, 64]
        # weight_maps: [K, 64, 64]
        # output: [K, 256]

        # [256, 64, 64] -> [256, 4096]
        image_embeddings = image_embeddings.flatten(1, 2)
        # [K, 64, 64] -> [K, 4096]
        weight_maps = weight_maps.flatten(1, 2)
        # [K, 4096] -> [K, 256]
        output = weight_maps @ image_embeddings.T
        # normalize
        output = output / (weight_maps.sum(-1, keepdim=True) + 1e-8)

        assert output.shape[0] == weight_maps.shape[0]
        assert output.shape[1] == image_embeddings.shape[0]

        return output

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        sam_segs_list: List[torch.FloatTensor],
        sam_ious_list: List[torch.FloatTensor],
        sam_iops_list: List[torch.FloatTensor],
        inference: bool = False,
        **kwargs,
    ):
        # image_embeddings = self.get_visual_embs(images)

        image_embeddings = self.get_dinov2_visual_embs(images)
        image_embeddings = self.model.lisa_dino_conv(image_embeddings)

        #import pdb; pdb.set_trace()
        
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # import pdb; pdb.set_trace()

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            # forward of LLaVA   
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        # import pdb; pdb.set_trace()

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        # import pdb; pdb.set_trace()

        # align sam proposals and pred_embeddings
        # sam_segs_list: list of (K, 64, 64), the length is batch_size, K is the number of SAM proposals
        # sam_ious_list: list of (K), the length is batch_size
        # pred_embeddings: list of (C, D) C is the number of conversations, D is the embedding dimension=256

        # upsample the image_embedding to 256x256
        # do not support fp16, because interpolate does not support fp16
        # see the disucssion here: https://github.com/pytorch/pytorch/issues/88536
        # first convert to float32
        origin_dtype = image_embeddings.dtype
        image_embeddings = image_embeddings.to(dtype = torch.float32)
        image_embeddings = F.interpolate(image_embeddings, size=(256, 256), mode='bilinear', align_corners=False)
        # convert back to original dtype
        image_embeddings = image_embeddings.to(dtype = origin_dtype)

        # mask pooling
        sam_segs_feature_list = []
        sam_pred_ious_list = []
        for batch_idx in range(len(sam_segs_list)):
            segs = sam_segs_list[batch_idx]
            segs_feature = self.mask_pooling(image_embeddings[batch_idx], segs)
            # use attention to update the mask feature, keep text embedding unchanged
            text_feature = pred_embeddings[batch_idx] # (C, D)
            # # add one dimension to text_feature （C, 1, D）
            text_feature = text_feature.unsqueeze(1)

            number_conversations = text_feature.shape[0]

            segs_feature = segs_feature.unsqueeze(0) # (1, K, D)
            if number_conversations > 0:
                # expand segs_feature to (C, K, D)
                segs_feature = segs_feature.expand(number_conversations, -1, -1)

            # import pdb; pdb.set_trace()    

            for layer in self.model.lisa_attention_layers:
                segs_feature, text_feature = layer(
                    queries=segs_feature,
                    keys=text_feature,
                )

            attn_out = self.model.lisa_final_attn(q=segs_feature, k=text_feature, v=text_feature)
            segs_feature = segs_feature + attn_out
            segs_feature = self.model.lisa_norm_final_attn(segs_feature)

            # use MLP to reduce the seg_features to 1 dimension
            sam_iou = self.model.lisa_iou_head(segs_feature) # (C, K, 1)
            sam_pred_ious_list.append(sam_iou) # (C, K, 1)

            segs_feature = self.model.lisa_embedding_head(segs_feature) # (C, K, D)
            sam_segs_feature_list.append(segs_feature)


        if inference:
            # during inference , C = 1
            pred_similarity = []
            for batch_idx in range(len(pred_embeddings)):
                pred_embedding = pred_embeddings[batch_idx] # (1, D)
                pred_embedding_normlized = pred_embedding / pred_embedding.norm(dim=-1, keepdim=True)
                sam_features = sam_segs_feature_list[batch_idx][0, :, :] # (K, D)
                sam_features_normlized = sam_features / sam_features.norm(dim=-1, keepdim=True)
                similarity = pred_embedding_normlized @ sam_features_normlized.T # (1, K)
                pred_similarity.append(similarity)

            pred_ious = []
            for batch_idx in range(len(sam_pred_ious_list)):
                sam_pred_ious = sam_pred_ious_list[batch_idx][0, :, :] # (K, 1)
                pred_ious.append(sam_pred_ious.T) # (1, K)

            return {
                "pred_similarity": pred_similarity,
                "gt_masks": masks_list,
                "pred_iou": pred_ious,
            }

        ce_loss = output.loss  # loss of LLaVA

        # compute align loss
        align_loss = 0.0 
        regression_loss = 0.0

        valid_batch = 0
        for batch_idx in range(len(sam_segs_feature_list)):
            segs_feature = sam_segs_feature_list[batch_idx]  # (K, D) D=256
            gt_iou = sam_ious_list[batch_idx]            # (R,K)
            gt_iop = sam_iops_list[batch_idx]            # (R,K)
            pred_iou = sam_pred_ious_list[batch_idx] 
            # # multiple conversations
            # assert gt_iou.shape[0] == pred_embeddings[batch_idx].shape[0], "number of rounds mismatch, gt_iou.shape: {}, pred_embeddings.shape: {}".format(gt_iou.shape, pred_embeddings[batch_idx].shape)
            # assert gt_iou.shape[0] != 0, "number of rounds = 0; gt_iou.shape: {}".format(gt_iou.shape)
            number_rounds = pred_embeddings[batch_idx].shape[0]

            align_loss_round = 0.0
            regression_losss_round = 0.0
            if number_rounds == 0:
                # throw an error 
                raise ValueError("number of rounds = 0; gt_iou.shape: {}".format(gt_iou.shape))

            for round_idx in range(number_rounds):
                gt_iou_round = gt_iou[round_idx] # (K)
                gt_iou_round = gt_iou_round.unsqueeze(1) # (K, 1)
                gt_iou_round = gt_iou_round.to(dtype = pred_iou.dtype)
                gt_iop_round = gt_iop[round_idx].unsqueeze(1).to(dtype = pred_iou.dtype) # (K, 1

                target_embedding = pred_embeddings[batch_idx][round_idx].unsqueeze(0)  # (1, D)
                # align_loss_round += sigmoid_align_loss(segs_feature, target_embedding, gt_iou_round, 
                #                                        self.model.temperature, self.model.bias)
                align_loss_round += softmax_align_loss(segs_feature[round_idx], target_embedding, gt_iou_round)
                regression_losss_round += iou_regression_loss(pred_iou[round_idx], gt_iop_round)
            
            if number_rounds > 0:
                valid_batch += 1

            align_loss +=  align_loss_round / (number_rounds + 1e-8)
            regression_loss += regression_losss_round / (number_rounds + 1e-8)

        if valid_batch > 0:
            align_loss = align_loss / valid_batch
            regression_loss = regression_loss / valid_batch

        # regression_loss = regression_loss.to(ce_loss.dtype)
        # loss = self.ce_loss_weight * ce_loss +  self.align_loss_weight * align_loss + self.regression_loss_weight * regression_loss
        
        ce_loss = ce_loss * self.ce_loss_weight
        align_loss = align_loss * self.align_loss_weight
        regression_loss = regression_loss * self.regression_loss_weight
        loss = ce_loss + align_loss + regression_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "align_loss": align_loss,
            "regression_loss": regression_loss,
        }


    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
