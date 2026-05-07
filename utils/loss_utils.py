import mpmath
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size, mdetr_interpolate


def build_target(args, gt_bbox, pred, device):
    batch_size = gt_bbox.size(0)
    num_scales = len(pred)
    coord_list, bbox_list = [], []
    for scale_ii in range(num_scales):
        this_stride = 32 // (2 ** scale_ii)
        grid = args.size // this_stride
        # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
        center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
        # Normalized by the image size
        coord = coord / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch_size):
        anch_ious = []
        for scale_ii in range(num_scales):
            this_stride = 32 // (2 ** scale_ii)
            grid = args.size // this_stride
            gw = coord_list[scale_ii][ii, 2]
            gh = coord_list[scale_ii][ii, 3]

            anchor_idxs = [x + 3 * scale_ii for x in [0, 1, 2]]
            anchors = [args.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (args.anchor_imsize / grid),
                               x[1] / (args.anchor_imsize / grid)) for x in anchors]

            gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.FloatTensor(
                np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))

            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n // 3

        best_grid = args.size // (32 / (2 ** best_scale))
        anchor_idxs = [x + 3 * best_scale for x in [0, 1, 2]]
        anchors = [args.anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [(x[0] / (args.anchor_imsize / best_grid), \
                           x[1] / (args.anchor_imsize / best_grid)) for x in anchors]

        gi = coord_list[best_scale][ii, 0].long()
        gj = coord_list[best_scale][ii, 1].long()
        tx = coord_list[best_scale][ii, 0] - gi.float()
        ty = coord_list[best_scale][ii, 1] - gj.float()
        gw = coord_list[best_scale][ii, 2]
        gh = coord_list[best_scale][ii, 3]
        tw = torch.log(gw / scaled_anchors[best_n % 3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n % 3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n % 3, :, gj, gi] = torch.stack(
            [tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].to(device)
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1. / 5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    num_scale = len(pred_list)
    batch_size = pred_list[0].size(0)

    pred_bbox = torch.zeros(batch_size, 4).to(device)
    gt_bbox = torch.zeros(batch_size, 4).to(device)
    for ii in range(batch_size):
        pred_bbox[ii, 0:2] = torch.sigmoid(
            pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
    loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
    loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
    loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(num_scale):
        pred_conf_list.append(pred_list[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
        gt_conf_list.append(target[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf


# Contrastive Loss
class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):

        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(pooled_image.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0
        return loss


# The below code is copied from transformers/models/clip/modeling_clip.py
# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def dice_loss(inputs, targets, num_boxes):
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
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes



def fuse_normalized_losses(norm_loss_dict, args):


    zero = None
    for v in norm_loss_dict.values():
        zero = v.sum() * 0.0
        break
    if zero is None:
        raise ValueError("norm_loss_dict is empty.")

    def get_loss(name):
        return norm_loss_dict.get(name, zero)

    loss_orig = get_loss("loss_orig_total_norm")
    loss_hand = get_loss("loss_hand_cls_norm")
    loss_kp   = get_loss("loss_kp_norm")
    loss_ray  = get_loss("loss_ray_norm")
    loss_angle = get_loss("loss_angle_norm")


    orig_total = getattr(args, "loss_orig_weight", 0.8)
    new_total  = getattr(args, "loss_new_weight", 0.2)


    use_hand_branch = getattr(args, "use_hand_branch", False)
    use_kp_branch   = getattr(args, "use_kp_branch", False)


    hand_ratio  = getattr(args, "loss_hand_ratio", 0.4) if use_hand_branch else 0.0
    kp_ratio    = getattr(args, "loss_kp_ratio", 0.3) if use_kp_branch else 0.0
    ray_ratio   = getattr(args, "loss_ray_ratio", 0.3) if use_kp_branch else 0.0
    angle_ratio = getattr(args, "loss_angle_ratio", 0.1) if use_kp_branch else 0.0

    new_loss = hand_ratio * loss_hand + kp_ratio * loss_kp + ray_ratio * loss_ray + angle_ratio * loss_angle
    total_loss = orig_total * loss_orig + new_total * new_loss

    weighted_loss_dict = {
        "loss_orig_total_weighted": orig_total * loss_orig,
        "loss_hand_weighted": new_total * hand_ratio * loss_hand,
        "loss_kp_weighted":   new_total * kp_ratio   * loss_kp,
        "loss_ray_weighted":  new_total * ray_ratio  * loss_ray,
        "loss_angle_weighted": new_total * angle_ratio * loss_angle,
        "loss_new_total":     new_total * new_loss,
        "loss_total":         total_loss,
    }

    return total_loss, weighted_loss_dict




def _one_ref_loss_original_unchanged(args, batch_pred, batch_target, tgt_mask, contrastive_loss,
                                     visu_sim=None, seg_mask=None,
                                     mim_pred=None, mim_labels=None, mim_vts_pred=None, mim_vts_labels=None,
                                     mlm_loss=None, mlm_sts_pred=None, mlm_sts_labels=None):

    batch_size = batch_pred.shape[0]
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(
        generalized_box_iou(
            xywh2xyxy(batch_pred),
            xywh2xyxy(batch_target)
        )
    )

    losses = {}
    if args.use_regress_box:
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

    if args.use_contrastive_loss:
        losses['loss_contrastive'] = contrastive_loss / num_boxes

    if args.use_box_mask_constraints or args.enable_dynamic_mim:
        coef_focal = 20.0
        coef_dice = 2.0
        patch_num = int(mpmath.sqrt(visu_sim.shape[-1]))
        obj_mask = mdetr_interpolate(tgt_mask.float(), (patch_num, patch_num), mode="nearest")[:, 0] > 0.5
        obj_mask = obj_mask.flatten(1).float()
        visu_sim_flat = visu_sim.flatten(1)

        losses['loss_mrm_focal'] = sigmoid_focal_loss(visu_sim_flat, obj_mask, num_boxes) * coef_focal
        losses['loss_mrm_dice'] = dice_loss(visu_sim_flat, obj_mask, num_boxes) * coef_dice

    if args.use_mask_loss:
        coef_focal = 20.0
        coef_dice = 2.0
        src_mask = mdetr_interpolate(seg_mask, size=tgt_mask.shape[-2:], mode="bilinear", align_corners=False)
        src_mask = src_mask.flatten(1)
        tgt_mask_flat = tgt_mask.flatten(1).float()

        losses['loss_seg_focal'] = sigmoid_focal_loss(src_mask, tgt_mask_flat, num_boxes) * coef_focal
        losses['loss_seg_dice'] = dice_loss(src_mask, tgt_mask_flat, num_boxes) * coef_dice

    if args.enable_ref_mlm and mlm_loss is not None:
        losses['loss_mlm'] = mlm_loss
        if args.enable_mlm_sts and mlm_sts_pred is not None and mlm_sts_labels is not None:
            if mlm_sts_pred.shape == mlm_sts_labels.shape:
                kl_loss = nn.KLDivLoss(reduction="batchmean")
                losses['loss_mlm_sts'] = kl_loss(
                    F.log_softmax(mlm_sts_pred, dim=-1),
                    F.softmax(mlm_sts_labels, dim=-1)
                )

    if args.enable_ref_mim and mim_pred is not None and mim_labels is not None:
        loss_fn = nn.CrossEntropyLoss()
        if isinstance(mim_pred, list):
            loss_1 = loss_fn(input=mim_pred[0], target=mim_labels)
            loss_2 = loss_fn(input=mim_pred[1], target=mim_labels)
            losses['loss_mim'] = loss_1 + loss_2
        else:
            losses['loss_mim'] = loss_fn(input=mim_pred, target=mim_labels)
            if args.enable_mim_vts and mim_vts_pred is not None and mim_vts_labels is not None:
                mim_vts_loss = F.l1_loss(mim_vts_pred, mim_vts_labels, reduction='none')
                losses['loss_mim_vts'] = mim_vts_loss.sum(dim=-1).mean()

    return losses


def one_ref_loss(args, batch_pred, batch_target, tgt_mask, contrastive_loss,
                 visu_sim=None, seg_mask=None,
                 mim_pred=None, mim_labels=None, mim_vts_pred=None, mim_vts_labels=None,
                 mlm_loss=None, mlm_sts_pred=None, mlm_sts_labels=None,
                 pred_kp=None,          # [B,4] = [root_x, root_y, tip_x, tip_y]
                 hand_logit=None,       # [B] or [B,1]
                 kp_root=None,          # [B,2]
                 kp_tip=None,           # [B,2]
                 kp_valid=None,         # [B]
                 is_positive=None):     # [B]

    losses = {}

    orig_loss_dict = _one_ref_loss_original_unchanged(
        args=args,
        batch_pred=batch_pred,
        batch_target=batch_target,
        tgt_mask=tgt_mask,
        contrastive_loss=contrastive_loss,
        visu_sim=visu_sim,
        seg_mask=seg_mask,
        mim_pred=mim_pred,
        mim_labels=mim_labels,
        mim_vts_pred=mim_vts_pred,
        mim_vts_labels=mim_vts_labels,
        mlm_loss=mlm_loss,
        mlm_sts_pred=mlm_sts_pred,
        mlm_sts_labels=mlm_sts_labels,
    )


    loss_orig_total_raw = sum(orig_loss_dict[k] for k in orig_loss_dict.keys())
    losses["loss_orig_total_raw"] = loss_orig_total_raw

    if getattr(args, "use_hand_branch", False) and hand_logit is not None and is_positive is not None:
        pred_hand = hand_logit.view(-1)
        target_hand = is_positive.float().view(-1)

        hand_cls_raw = F.binary_cross_entropy_with_logits(pred_hand, target_hand)
        losses["loss_hand_cls_raw"] = hand_cls_raw

    else:
        losses["loss_hand_cls_raw"] = batch_pred.sum() * 0.0

    #if pred_kp is not None and kp_root is not None and kp_tip is not None and kp_valid is not None:
    if getattr(args, "use_kp_branch", False) and pred_kp is not None and kp_root is not None and kp_tip is not None and kp_valid is not None:
        valid_mask = kp_valid.bool().view(-1)
        if is_positive is not None:
            valid_mask = valid_mask & is_positive.bool().view(-1)

        if valid_mask.any():
            gt_kp = torch.cat([kp_root, kp_tip], dim=-1)  # [B,4]

            kp_l1 = F.smooth_l1_loss(pred_kp, gt_kp, reduction='none').mean(dim=-1)  # [B]
            kp_point_raw = kp_l1[valid_mask].mean()

            pred_root = pred_kp[:, 0:2]   # [B,2]
            pred_tip = pred_kp[:, 2:4]    # [B,2]
            gt_root = kp_root             # [B,2]
            gt_tip = kp_tip               # [B,2]

            pred_ray_len = torch.norm(pred_tip - pred_root, dim=-1)  # [B]
            gt_ray_len = torch.norm(gt_tip - gt_root, dim=-1)        # [B]

            kp_len_l1 = F.smooth_l1_loss(pred_ray_len, gt_ray_len, reduction='none')  # [B]
            kp_len_raw = kp_len_l1[valid_mask].mean()

            kp_raw = kp_point_raw + 1* kp_len_raw
            losses["loss_kp_raw"] = kp_raw

        else:
            losses["loss_kp_raw"] = pred_kp.sum() * 0.0
    else:
        losses["loss_kp_raw"] = batch_pred.sum() * 0.0

    
    if getattr(args, "use_kp_branch", False) and pred_kp is not None and batch_target is not None and kp_valid is not None and is_positive is not None:
        valid = kp_valid.bool().view(-1) & is_positive.bool().view(-1)

        if valid.any():
            r = pred_kp[:, 0:2]   # predicted root
            t = pred_kp[:, 2:4]   # predicted tip

            # predicted ray direction: root -> tip
            d_rt = t - r

            # GT bbox center
            #print(batch_target[0])
            c_gt = batch_target[:, 0:2]

            eps = 1e-6
            d_rt_norm = torch.norm(d_rt, dim=-1)

            # tip -> GT center
            d_tc = c_gt - t
            d_tc_norm = torch.norm(d_tc, dim=-1)

            good_dir_rt = d_rt_norm > 1e-4
            good_dir_tc = d_tc_norm > 1e-4
            valid2 = valid & good_dir_rt & good_dir_tc

            if valid2.any():
                u_rt = d_rt / (d_rt_norm.unsqueeze(-1) + eps)   # [B, 2]
                v = c_gt - r                                    # [B, 2]

                s = (v * u_rt).sum(dim=-1)                      # [B]
                s_clamped = torch.clamp(s, min=0.0)             
                p = r + s_clamped.unsqueeze(-1) * u_rt          # [B, 2]

                dist = torch.norm(c_gt - p, dim=-1)             # [B]
                ray_raw = dist[valid2].mean()
                losses["loss_ray_raw"] = ray_raw

                u_tc = d_tc / (d_tc_norm.unsqueeze(-1) + eps)   # [B, 2]
                cos_sim = (u_rt * u_tc).sum(dim=-1)             # [B]
                cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)

                angle_raw = (1.0 - cos_sim[valid2]).mean()
                losses["loss_angle_raw"] = angle_raw
            else:
                losses["loss_ray_raw"] = pred_kp.sum() * 0.0
                losses["loss_angle_raw"] = pred_kp.sum() * 0.0
        else:
            losses["loss_ray_raw"] = pred_kp.sum() * 0.0
            losses["loss_angle_raw"] = pred_kp.sum() * 0.0
    else:
        zero_base = pred_kp.sum() * 0.0 if pred_kp is not None else batch_target.sum() * 0.0
        losses["loss_ray_raw"] = zero_base
        losses["loss_angle_raw"] = zero_base

    return losses