import torch
import torch.nn.functional as F

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def softmax_align_loss(proposal_embeds: torch.tensor, target_embed: torch.tensor, gt_ious: torch.tensor,temperature: float = 0.05):
    """
    Align the similarity with the ground truth iou, because iou is not integers, we
    actually have soft labels for the similarity. Instead of computing the cross entropy
    loss, we can compute the KL divergence between the similarity and the ground truth.
    
    The loss is based on RegionClip.
    https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/detectron2/modeling/meta_arch/clip_rcnn.py#L587

    proposal_embeds: (K, D)
    target_embed: (1, D)
    gt_ious: (K, 1)
    """

    # normalize the proposal_embeds and target_embed
    proposal_embeds = proposal_embeds / proposal_embeds.norm(dim=-1, keepdim=True)
    target_embed = target_embed / target_embed.norm(dim=-1, keepdim=True)

    # compute similarity scores
    sim_scores = proposal_embeds @ target_embed.t()  # (K, 1)
    sim_scores_temp = sim_scores / temperature
    gt_iou_temp = gt_ious / temperature
    # normalize to distribution
    sim_dis = F.softmax(sim_scores_temp, dim=0)  # (K, 1)
    gt_dis = F.softmax(gt_iou_temp, dim=0)  # (K, 1)

    # for KL divergence, the input should be log-probability, the target is probability (when log_target==False)
    # Be Careful: loss domainted by negative samples, current use sum instead of mean
    loss = F.kl_div(sim_dis.log(), gt_dis, reduction="sum")

    return loss

def iou_regression_loss(pred_ious: torch.tensor, gt_ious: torch.tensor, weighted: bool = True):
    """
    pred_ious: (K, 1)
    gt_ious: (K, 1)
    """
    if not weighted:
        loss = F.mse_loss(pred_ious, gt_ious, reduction="sum")
    else:
        loss = F.mse_loss(pred_ious, gt_ious, reduction="none")
        weight = torch.exp(gt_ious - 1.0)
        loss = loss * weight 
        loss = loss.mean() * 50.0  # scale the loss as if all sample has 50 proposals
    return loss


def sigmoid_align_loss(proposal_embeds: torch.tensor, target_embed: torch.tensor, gt_ious: torch.tensor,temperature: torch.tensor = 0.1, bias: torch.tensor = 0.0):
    """
    Sigmoid loss for contrastive learning.
    From Paper: Sigmoid Loss for Language Image Pre-Training (https://arxiv.org/abs/2303.15343)

    proposal_embeds: (K, D)
    target_embed: (1, D)
    gt_ious: (K, 1)

    temperature and bias are learnable parameters
    """

    sigmoid_layer = torch.nn.Sigmoid()

    t = torch.exp(temperature)
    b = bias

    # normalize the proposal_embeds and target_embed
    proposal_embeds = proposal_embeds / proposal_embeds.norm(dim=-1, keepdim=True)
    target_embed = target_embed / target_embed.norm(dim=-1, keepdim=True)

    logits = proposal_embeds @ target_embed.t() * t + b # (K, 1)
    # logits = sigmoid_layer(logits)

    # (K, 1) range from -1 to 1, we treat iou=0 as pure negative sample, iou=1 as pure positive sample, iou=0.5 as neutral sample
    # iou=0.5 as neutral may not be a good idea (maybe too high), but we can try
    labels = gt_ious * 2 - 1.0 

    loss = -1.0 * torch.log(sigmoid_layer(logits * labels) + 1e-8 ).sum()

    # loss = F.binary_cross_entropy_with_logits(logits, gt_ious, reduction="sum")

    return loss


def too_simple_to_believe_align_loss(proposal_embeds: torch.tensor, target_embed: torch.tensor, gt_ious: torch.tensor):
    """
    proposal_embeds: (K, D)
    target_embed: (1, D)
    gt_ious: (K, 1) 
    """

    # scale to -1 ~ 1
    label = gt_ious * 2.0 - 1.0

    # normalize the proposal_embeds and target_embed
    proposal_embeds = proposal_embeds / proposal_embeds.norm(dim=-1, keepdim=True)
    target_embed = target_embed / target_embed.norm(dim=-1, keepdim=True) 
    
    # the range of cosine similarity is [-1, 1]
    similarity = proposal_embeds @ target_embed.t()  # (K, 1)

    loss = F.l1_loss(similarity, label, reduction="sum")

    return loss

if __name__ == "__main__":
    # test contrastive_align_loss
    pass
