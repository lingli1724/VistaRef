# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from utils.box_utils import xywh2xyxy
import numpy as np


def get_module_grad_norm(module):
    total_norm_sq = 0.0
    has_grad = False
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2
            has_grad = True
    total_norm = total_norm_sq ** 0.5
    return total_norm, has_grad

def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, start_steps: int, max_norm: float = 0,loss_normalizer=None):
    if loss_normalizer is None:
        raise ValueError("loss_normalizer is None. Please create it in main() and pass it into train_one_epoch().")
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // args.update_freq
        global_step = start_steps + step  # global training iteration

        # ============================================================
        # [MODIFICATION BEGIN] robust batch unpacking
        # old batch: (img_data, text_data, target, obj_mask)
        # new batch: (img_data, text_data, target, obj_mask, kp_root, kp_tip, kp_valid, is_positive)
        # ============================================================
        if not isinstance(batch, (list, tuple)):
            raise TypeError(f"batch must be list/tuple, got {type(batch)}")

        if len(batch) == 4:
            img_data, text_data, target, obj_mask = batch
            kp_root = kp_tip = kp_valid = is_positive = None
            phrase = None
            bbox_ori = None
        elif len(batch) >= 8:
            img_data, text_data, target, obj_mask, kp_root, kp_tip, kp_valid, is_positive = batch[:8]
            phrase = batch[8] if len(batch) > 8 else None
            bbox_ori = batch[9] if len(batch) > 9 else None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        # ============================================================
        # [MODIFICATION END]
        # ============================================================

        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        obj_mask = obj_mask.to(device)

        if kp_root is not None:
            kp_root = kp_root.to(device)
        if kp_tip is not None:
            kp_tip = kp_tip.to(device)
        if kp_valid is not None:
            kp_valid = kp_valid.to(device)
        if is_positive is not None:
            is_positive = is_positive.to(device)

        # ============================================================
        # [NEW] batch-level supervision statistics
        # ============================================================
        if is_positive is not None:
            num_pos = int(is_positive.sum().item())
            num_neg = int((is_positive == 0).sum().item())
        else:
            num_pos = -1
            num_neg = -1

        if kp_valid is not None:
            num_kp_valid = int(kp_valid.sum().item())
        else:
            num_kp_valid = -1

        if kp_valid is not None and is_positive is not None:
            ray_valid_mask = kp_valid.bool().view(-1) & is_positive.bool().view(-1)
            num_ray_valid = int(ray_valid_mask.sum().item())
        else:
            ray_valid_mask = None
            num_ray_valid = -1

 
        print_freq_1=50
        model_outputs = model(
            img_data.tensors,
            img_data.mask,
            text_data,
            global_step=global_step,
            training=True,
            debug_forward=(getattr(args, "debug_loss", False) and data_iter_step % print_freq_1 == 0)
        )

        if not isinstance(model_outputs, (list, tuple)):
            raise TypeError(f"model output must be tuple/list, got {type(model_outputs)}")

        if len(model_outputs) == 9:
            pred_box, contrastive_loss, visu_sim, seg_mask, mlm_loss, mlm_acc, mlm_sts_pred, mim_pred, mim_vts_pred = model_outputs
            pred_kp = None
            hand_logit = None
        elif len(model_outputs) >= 11:
            pred_box, contrastive_loss, visu_sim, seg_mask, mlm_loss, mlm_acc, mlm_sts_pred, mim_pred, mim_vts_pred, pred_kp, hand_logit = model_outputs[:11]
        else:
            raise ValueError(f"Unexpected model output length: {len(model_outputs)}")

        hand_prob_mean = 0.0
        hand_prob_pos_mean = 0.0
        hand_prob_neg_mean = 0.0
        ray_len_mean = 0.0
        ray_len_std = 0.0

        if hand_logit is not None:
            hand_prob = torch.sigmoid(hand_logit.view(-1))
            hand_prob_mean = float(hand_prob.mean().item())

            if is_positive is not None:
                pos_mask = is_positive.bool().view(-1)
                neg_mask = ~pos_mask

                if pos_mask.any():
                    hand_prob_pos_mean = float(hand_prob[pos_mask].mean().item())
                if neg_mask.any():
                    hand_prob_neg_mean = float(hand_prob[neg_mask].mean().item())

        if pred_kp is not None:
            pred_root = pred_kp[:, 0:2]
            pred_tip = pred_kp[:, 2:4]
            ray_len = torch.norm(pred_tip - pred_root, dim=-1)
            ray_len_mean = float(ray_len.mean().item())
            ray_len_std = float(ray_len.std().item())

       
        raw_loss_dict = loss_utils.one_ref_loss(
            args=args,
            batch_pred=pred_box,
            batch_target=target,
            tgt_mask=obj_mask,
            contrastive_loss=contrastive_loss,
            visu_sim=visu_sim,
            seg_mask=seg_mask,
            mim_pred=mim_pred,
            mim_labels=None,
            mim_vts_pred=mim_vts_pred,
            mim_vts_labels=None,
            mlm_loss=mlm_loss,
            mlm_sts_pred=mlm_sts_pred,
            mlm_sts_labels=None,
            pred_kp=pred_kp,
            hand_logit=hand_logit,
            kp_root=kp_root,
            kp_tip=kp_tip,
            kp_valid=kp_valid,
            is_positive=is_positive,
        )

        # --------------------------------------------------
        # reduce RAW losses across GPUs for logging / EMA stats
        # use detached tensors for stats
        # --------------------------------------------------
        raw_loss_dict_detached = {k: v.detach() for k, v in raw_loss_dict.items()}
        raw_loss_dict_reduced = utils.reduce_dict(raw_loss_dict_detached)

        # update EMA stats using reduced raw losses
        ema_input = {k: v for k, v in raw_loss_dict_reduced.items() if k.endswith("_raw")}
        loss_normalizer.update(ema_input)

        # normalize LOCAL raw losses (with graph)
        raw_only_local = {k: v for k, v in raw_loss_dict.items() if k.endswith("_raw")}
        norm_loss_dict = loss_normalizer.normalize(raw_only_local)

        # # fuse normalized losses
        losses, weighted_loss_dict = loss_utils.fuse_normalized_losses(norm_loss_dict, args)

        norm_loss_dict_detached = {k: v.detach() for k, v in norm_loss_dict.items()}
        weighted_loss_dict_detached = {k: v.detach() for k, v in weighted_loss_dict.items()}

        norm_loss_dict_reduced = utils.reduce_dict(norm_loss_dict_detached)
        weighted_loss_dict_reduced = utils.reduce_dict(weighted_loss_dict_detached)

        loss_value = weighted_loss_dict_reduced["loss_total"].item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print("raw_loss_dict_reduced =", raw_loss_dict_reduced)
            print("norm_loss_dict_reduced =", norm_loss_dict_reduced)
            print("weighted_loss_dict_reduced =", weighted_loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        # ============================================================
        # [NEW] gradient statistics after backward
        # ============================================================
        grad_norm_bbox_head = 0.0
        grad_norm_kp_head = 0.0
        grad_norm_hand_head = 0.0

        model_for_grad = model.module if hasattr(model, "module") else model

        if hasattr(model_for_grad, "bbox_embed"):
            grad_norm_bbox_head, _ = get_module_grad_norm(model_for_grad.bbox_embed)
        if hasattr(model_for_grad, "kp_head"):
            grad_norm_kp_head, _ = get_module_grad_norm(model_for_grad.kp_head)
        if hasattr(model_for_grad, "hand_cls_head"):
            grad_norm_hand_head, _ = get_module_grad_norm(model_for_grad.hand_cls_head)
        # ============================================================

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if data_iter_step % print_freq == 0:
            print(
                f"[TrainDebug][E{epoch}][I{data_iter_step}] "
                f"lr={optimizer.param_groups[0]['lr']:.8e} "
                f"total={loss_value:.6f} "
                f"main={weighted_loss_dict_reduced.get('loss_main', torch.tensor(0.0)).item():.6f} "
                f"aux={weighted_loss_dict_reduced.get('loss_aux', torch.tensor(0.0)).item():.6f}"
            )

            print(
                f"[BatchStat][E{epoch}][I{data_iter_step}] "
                f"num_pos={num_pos} num_neg={num_neg} "
                f"num_kp_valid={num_kp_valid} num_ray_valid={num_ray_valid}"
            )

            print(
                f"[PredStat][E{epoch}][I{data_iter_step}] "
                f"hand_prob_mean={hand_prob_mean:.6f} "
                f"hand_prob_pos_mean={hand_prob_pos_mean:.6f} "
                f"hand_prob_neg_mean={hand_prob_neg_mean:.6f} "
                f"ray_len_mean={ray_len_mean:.6f} "
                f"ray_len_std={ray_len_std:.6f}"
            )

            raw_msg = f"[RawLossStat][E{epoch}][I{data_iter_step}]"
            for k in sorted(raw_loss_dict_reduced.keys()):
                raw_msg += f"  {k}={raw_loss_dict_reduced[k].item():.6f}"
            print(raw_msg)

            norm_msg = f"[NormLossStat][E{epoch}][I{data_iter_step}]"
            for k in sorted(norm_loss_dict_reduced.keys()):
                norm_msg += f"  {k}={norm_loss_dict_reduced[k].item():.6f}"
            print(norm_msg)

            weighted_msg = f"[WeightedLossStat][E{epoch}][I{data_iter_step}]"
            for k in sorted(weighted_loss_dict_reduced.keys()):
                weighted_msg += f"  {k}={weighted_loss_dict_reduced[k].item():.6f}"
            print(weighted_msg)

            if getattr(args, "debug_loss", False):
                ema_msg = f"[EMALossStat][E{epoch}][I{data_iter_step}]"
                for k in sorted(loss_normalizer.stats.keys()):
                    ema_msg += f"  {k}={loss_normalizer.stats[k].item():.6f}"
                print(ema_msg)

            print(
                f"[GradStat][E{epoch}][I{data_iter_step}] "
                f"grad_norm_bbox_head={grad_norm_bbox_head:.6e} "
                f"grad_norm_kp_head={grad_norm_kp_head:.6e} "
                f"grad_norm_hand_head={grad_norm_hand_head:.6e}"
            )

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # raw losses
        for k, v in raw_loss_dict_reduced.items():
            metric_logger.update(**{k: v.item()})

        # normalized losses
        for k, v in norm_loss_dict_reduced.items():
            metric_logger.update(**{k: v.item()})

        # weighted / fused losses
        for k, v in weighted_loss_dict_reduced.items():
            metric_logger.update(**{k: v.item()})


        if num_pos >= 0:
            metric_logger.update(num_pos=float(num_pos))
        if num_neg >= 0:
            metric_logger.update(num_neg=float(num_neg))
        if num_kp_valid >= 0:
            metric_logger.update(num_kp_valid=float(num_kp_valid))
        if num_ray_valid >= 0:
            metric_logger.update(num_ray_valid=float(num_ray_valid))

        metric_logger.update(hand_prob_mean=hand_prob_mean)
        metric_logger.update(hand_prob_pos_mean=hand_prob_pos_mean)
        metric_logger.update(hand_prob_neg_mean=hand_prob_neg_mean)
        metric_logger.update(ray_len_mean=ray_len_mean)
        metric_logger.update(ray_len_std=ray_len_std)

        metric_logger.update(grad_norm_bbox_head=grad_norm_bbox_head)
        metric_logger.update(grad_norm_kp_head=grad_norm_kp_head)
        metric_logger.update(grad_norm_hand_head=grad_norm_hand_head)
        

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


"""
   The core training code is implemented here, which alternately models MIM and MLM.
   Implemented by Linhui Xiao.
     2024-01-10
"""
def train_one_epoch_with_mrefm(args, model: torch.nn.Module, vqkd: torch.nn.Module, data_loader: Iterable,
                               optimizer: torch.optim.Optimizer, device: torch.device,
                               epoch: int, start_steps: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 11  # ori: 10

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // args.update_freq
        global_step = start_steps + step  # global training iteration

        if global_step % 2 == 0:
            enable_ref_mim = True
            enable_ref_mlm = False
        else:
            enable_ref_mim = False
            enable_ref_mlm = True

        img_data, text_data, target, obj_mask, mim_img, mim_mask_pos, mim_vts_labels, mlm_sts_labels = batch
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        obj_mask = obj_mask.to(device)  # obj_mask shape:  torch.Size([96, 1, 224, 224])
        mim_img = mim_img.to(device)  # non_blocking=True)
        mim_mask_pos = mim_mask_pos.to(device)  # non_blocking=True)
        mim_vts_labels = mim_vts_labels.to(device)  # torch.Size([64, 576, 4])
        """ If the original text is passed in, uncomment out the following code. """
        # text_data = text_data.to(device)

        if enable_ref_mim:
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                input_ids = vqkd.get_codebook_indices(mim_img)  # Tokenize the original image, torch.Size([24, 24, 24])
                bool_masked_pos = mim_mask_pos.flatten(1).to(torch.bool)  # numpy to torch, torch.Size([24, 576])
                mim_labels = input_ids[bool_masked_pos]  # Get the ID based on the mask, shape: torch.Size([5520]), 24*230=5520
        else:
            bool_masked_pos, mim_labels = None, None

        # model forward
        pred_box, contrastive_loss, visu_sim, seg_mask, mlm_loss, mlm_acc, mlm_sts_pred, mim_pred, mim_vts_pred = \
            model(img_data.tensors, img_data.mask, text_data, global_step=global_step, mim_masked_pos=bool_masked_pos,
                  obj_mask=obj_mask, enable_ref_mim=enable_ref_mim, enable_ref_mlm=enable_ref_mlm, training=True)
        # The `loss_dict` is a dictionary that contains `l1_smooth` and `giou`.
        loss_dict = loss_utils.one_ref_loss(args, pred_box, target, obj_mask, contrastive_loss, visu_sim, seg_mask,
                                            mim_pred, mim_labels, mim_vts_pred, mim_vts_labels,
                                            mlm_loss, mlm_sts_pred, mlm_sts_labels)

        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:  # The default value of max_norm is 0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        
        img_data, text_data, target, tgt_mask = batch[:4]

        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)
        """ If the original text is passed in, uncomment out the following code. """
        # text_data = text_data.to(device)


        out = model(img_data.tensors, img_data.mask, text_data)
        pred_box, seg_mask, img_cls, text_cls = out[:4]   
        
        miou, accu, mask_iou_list, I_list, U_list = eval_utils.trans_vg_eval_val(args, pred_box, target, seg_mask, tgt_mask)

        metric_logger.update_v2('box_miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('box_accu', accu, batch_size)
        if mask_iou_list is not None:
            metric_logger.update_v2('seg_miou', torch.mean(mask_iou_list), batch_size)

        if args.use_mask_loss:
            metric_logger.update_v2('accu', torch.mean(mask_iou_list), batch_size)
        else:
            metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []

    pred_mask_list = []
    gt_mask_list = []

    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)
        """ If the original text is passed in, uncomment out the following code. """
        # text_data = text_data.to(device)

        pred_box, seg_mask, img_cls, text_cls = model(img_data.tensors, img_data.mask, text_data)

        pred_box_list.append(pred_box.cpu())
        gt_box_list.append(target.cpu())

        pred_mask_list.append(seg_mask.cpu())
        gt_mask_list.append(tgt_mask.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    pred_masks = torch.cat(pred_mask_list, dim=0)
    gt_masks = torch.cat(gt_mask_list, dim=0)

    total_num = gt_boxes.shape[0]
    accu_num, iou, mask_iou_list, I_list, U_list = eval_utils.trans_vg_eval_test(args, pred_boxes, gt_boxes, pred_masks, gt_masks)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    if args.use_mask_loss:
        acc_mask_iou = torch.sum(mask_iou_list, dim=0)
        mask_result_tensor = torch.tensor([acc_mask_iou, total_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)
    if args.use_mask_loss:
        dist.all_reduce(mask_result_tensor)

    box_accuracy = float(result_tensor[0]) / float(result_tensor[1])

    if args.use_mask_loss:
        seg_miou = float(mask_result_tensor[0]) / float(mask_result_tensor[1])
        print("segmentation mIoU: ", seg_miou)
        seg_oiou = float(torch.sum(I_list, dim=0)) / float(torch.sum(U_list, dim=0))
        print("segmentation oIoU: ", seg_oiou)
        return seg_miou

    return box_accuracy


@torch.no_grad()
def evaluate_hivg(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    text_list = []

    pred_mask_list = []
    gt_mask_list = []

    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        # text_data = text_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)
        """Core model calculation"""
        output, _, _, token_sim, seg_mask = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

        pred_mask_list.append(seg_mask.cpu())
        gt_mask_list.append(tgt_mask.cpu())

        for text_i in text_data:
            text_list.append(text_i)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)

    pred_masks = torch.cat(pred_mask_list, dim=0)
    gt_masks = torch.cat(gt_mask_list, dim=0)

    total_num = gt_boxes.shape[0]
    accu_num, iou, mask_iou_list, I_list, U_list = eval_utils.trans_vg_eval_test(args, pred_boxes, gt_boxes, pred_masks, gt_masks)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    if args.use_mask_loss:
        acc_mask_iou = torch.sum(mask_iou_list, dim=0)
        mask_result_tensor = torch.tensor([acc_mask_iou, total_num]).to(device)


    """" Statistics the result with different text length """

    # statistic_diff_length_acc = True
    statistic_diff_length_acc = False
    # only can be used in one GPU，Using multiple cards will only print the result of a single card.
    if statistic_diff_length_acc:
        assert len(text_list) == iou.shape[0]
        count_for_len_in_1_to_5 = [0, 0]
        count_for_len_in_6_to_7 = [0, 0]
        count_for_len_in_8_to_10 = [0, 0]
        count_for_len_in_11_plus = [0, 0]
        for i in range(len(text_list)):
            len_i = len(text_list[i].split(" "))
            iou_i = iou[i]
            if (len_i >= 1) and (len_i <= 5):
                count_for_len_in_1_to_5[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_1_to_5[0] += 1
            elif (len_i >= 6) and (len_i <= 7):
                count_for_len_in_6_to_7[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_6_to_7[0] += 1
            elif (len_i >= 8) and (len_i <= 10):
                count_for_len_in_8_to_10[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_8_to_10[0] += 1
            elif (len_i >= 11):
                count_for_len_in_11_plus[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_11_plus[0] += 1

        print("acc in length  1-5: ", count_for_len_in_1_to_5, ", ",
              count_for_len_in_1_to_5[0] / count_for_len_in_1_to_5[1])
        print("acc in length  6-7: ", count_for_len_in_6_to_7, ", ",
              count_for_len_in_6_to_7[0] / count_for_len_in_6_to_7[1])
        print("acc in length 8-10: ", count_for_len_in_8_to_10, ", ",
              count_for_len_in_8_to_10[0] / count_for_len_in_8_to_10[1])
        print("acc in length  11+: ", count_for_len_in_11_plus, ", ",
              count_for_len_in_11_plus[0] / count_for_len_in_11_plus[1])

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)
    if args.use_mask_loss:
        dist.all_reduce(mask_result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    print("accuracy2: ", accuracy)
    if args.use_mask_loss:
        miou = float(mask_result_tensor[0]) / float(mask_result_tensor[1])
        print("segmentation miou: ", miou)

    return accuracy


@torch.no_grad()
def evaluate_clip_vg(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, obj_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        output, _, _, _, seg_mask = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])

    return accuracy

