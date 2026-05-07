import os
import time
import json
import math
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from datasets import build_dataset
import utils.eval_utils as eval_utils  # 你贴的 eval_utils.py

# OneRef required imports (不要删)
from timm.models import create_model
import models.VistaRef_model as VistaRef_model  # noqa: F401
import models.modeling_vqkd as modeling_vqkd  # noqa: F401

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF



def get_args_parser():
    parser = argparse.ArgumentParser('OneRef Args', add_help=False)
    parser.add_argument('--sup_type', default='full', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--lr_exponential', default=0.9, type=float, help='lr exponential')
    parser.add_argument('--clip_max_norm', default=0., type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true', help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true', help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true', help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true', help="If true, use random translate augmentation")
    # BEiT-3 Args
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task', type=str, required=True,
                        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps',
                                 'imagenet', 'grounding'], help='Name of task to fine-tuning')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--checkpoint_activations', action='store_true', default=None,
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--sentencepiece_model', type=str,
                        default='/hdd/lhxiao/beit3/checkpoint/beit3.spm',
                        help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--eval_batch_size', default=None, type=int)
    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--enable_seg_mask', action='store_true', help="If true, use segmentation mask, otherwise use box mask.")
    parser.add_argument('--frozen_backbone', action='store_true', help="If true, frozen BEiT-3", default=False)
    parser.add_argument('--use_contrastive_loss', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--use_box_mask_constraints', action='store_true', help="If true, use box mask constraints")
    parser.add_argument('--use_mask_loss', action='store_true', help="If true, use segmentation loss")
    parser.add_argument('--use_regress_box', action='store_true', help="If true, enable regress box loss")
    parser.add_argument('--enable_ref_mlm', action='store_true', help="If true, use mlm loss")
    parser.add_argument('--enable_ref_mim', action='store_true', help="If true, use mim loss")
    parser.add_argument('--enable_dynamic_mim', action='store_true', help="If true, use mim loss")
    parser.add_argument('--mim_mask_ratio', type=float, default=0.4)

    parser.add_argument('--text_mask_prob', type=float, default=0.4)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--update_freq', default=1, type=int)

    # mim pretraining
    # cls-pretraining settings
    parser.add_argument('--early_layers', default=9, type=int, help='early_layers, default 9 for base and 21 for large')
    parser.add_argument('--head_layers', default=2, type=int, help='head_layers')
    parser.add_argument('--mim_mid_layer', default=0, type=int, help='mim_mid_layer,set 0 or 9')
    parser.add_argument('--shared_lm_head', default=True, type=utils.bool_flag, help='head_layers')

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='hidden dimension of codebook')
    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str, default="/hdd/lhxiao/beit2/checkpoint/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth")
    parser.add_argument("--tokenizer_model", type=str, default="vqkd_encoder_base_decoder_3x768x12_clip")

    parser.add_argument('--num_mask_patches', default=75, type=int, help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=224, type=int, help='image size')
    """ embedding size"""
    parser.add_argument('--emb_size', default=512, type=int, help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=512, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./data/image_data/', help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='./data/pseudo_samples/',  help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str, help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument("--egopoint_jsonl", type=str, default="", help="path to EgoPoint jsonl")
    parser.add_argument('--max_query_len', default=77, type=int, help='maximum time steps (lang length) per batch')
    # Prompt Engineering: "{pseudo_query}" denote without using prompt
    #                     "{pseudo_query}" or using "find the region that corresponds to the description {pseudo_query}"
    parser.add_argument('--prompt', type=str, default='{pseudo_query}', help="Prompt template")
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--retrain', default='', help='retrain from checkpoint')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # evalutaion options
    parser.add_argument('--eval_set', default='test', type=str)  # 'testA', 'testB', 'val'
    parser.add_argument('--eval_model', default='', type=str)

    parser.add_argument("--hand_thresh", type=float, default=0.5, help="threshold on sigmoid(hand_logit) for negative/positive decision")

    parser.add_argument(
        "--iou_mode",
        type=str,
        default="pos_only",
        choices=["pos_only", "penalize_neg_fp"],
        help="IoU evaluation mode: "
            "pos_only = compute IoU on GT positives only; "
            "penalize_neg_fp = additionally count IoU=0 for GT negatives predicted as positive."
    )

    parser.add_argument(
        '--eval_with_box_zero_as_neg',
        action='store_true',
        help='If true, for old models without hand branch, treat pred_box == [0,0,0,0] as predicted negative.'
    )

    parser.add_argument('--visualize', action='store_true', help='save visualization results')
    parser.add_argument('--num_vis', default=30, type=int, help='number of samples to visualize')
    parser.add_argument('--vis_dir', default='vis_results', type=str, help='directory to save visualization images')

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')
    parser.add_argument("--use_hand_branch", type=str2bool, default=True)
    parser.add_argument("--use_kp_branch", type=str2bool, default=True)
    parser.add_argument("--use_ocal_module", type=str2bool, default=True)


    return parser


def _model_config(args):
    if not args.model.endswith(args.task):
        if args.task == "grounding":
            return f"{args.model}_grounding"
        return f"{args.model}_{args.task}"
    return args.model

def denormalize_image(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    img_tensor: [3,H,W], normalized tensor
    return: PIL.Image
    """
    img = img_tensor.detach().cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    return TF.to_pil_image(img)


def xywh_norm_to_xyxy_abs(box_xywh, image_size=640):
    """
    box_xywh: Tensor/list [cx, cy, w, h], normalized to [0,1]
    return: [x1, y1, x2, y2] in pixels on padded 640x640 image
    """
    if torch.is_tensor(box_xywh):
        box_xywh = box_xywh.detach().cpu().float()

    cx, cy, w, h = box_xywh.tolist()
    cx *= image_size
    cy *= image_size
    w *= image_size
    h *= image_size

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    x1 = max(0, min(image_size - 1, x1))
    y1 = max(0, min(image_size - 1, y1))
    x2 = max(0, min(image_size - 1, x2))
    y2 = max(0, min(image_size - 1, y2))
    return [x1, y1, x2, y2]


def kp_norm_to_abs(kp_xy, image_size=640):
    """
    kp_xy: Tensor/list [x, y], normalized to [0,1]
    return: [x, y] in pixels on padded 640x640 image
    """
    if torch.is_tensor(kp_xy):
        kp_xy = kp_xy.detach().cpu().float()
    x, y = kp_xy.tolist()
    x *= image_size
    y *= image_size
    x = max(0, min(image_size - 1, x))
    y = max(0, min(image_size - 1, y))
    return [x, y]


def hand_patch_mask_to_bbox(hand_patch_mask, image_size=640):
    """
    hand_patch_mask: [N] bool or 0/1, patch-level hand mask
    将 patch mask 转成近似 hand bbox
    return: [x1, y1, x2, y2] or None
    """
    if hand_patch_mask is None:
        return None

    if torch.is_tensor(hand_patch_mask):
        mask = hand_patch_mask.detach().cpu().bool().view(-1)
    else:
        mask = torch.tensor(hand_patch_mask).bool().view(-1)

    if mask.sum() == 0:
        return None

    n = mask.numel()
    patch_num = int(math.sqrt(n))
    if patch_num * patch_num != n:
        return None

    mask2d = mask.view(patch_num, patch_num)
    ys, xs = torch.where(mask2d)

    if len(xs) == 0:
        return None

    patch_size = image_size / patch_num

    x1 = xs.min().item() * patch_size
    y1 = ys.min().item() * patch_size
    x2 = (xs.max().item() + 1) * patch_size
    y2 = (ys.max().item() + 1) * patch_size

    x1 = max(0, min(image_size - 1, x1))
    y1 = max(0, min(image_size - 1, y1))
    x2 = max(0, min(image_size - 1, x2))
    y2 = max(0, min(image_size - 1, y2))

    return [x1, y1, x2, y2]


def draw_box(draw, box, color, width=3, text=None):
    if box is None:
        return
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if text is not None:
        draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color)


def draw_point(draw, pt, color, r=4, text=None):
    if pt is None:
        return
    x, y = pt
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    if text is not None:
        draw.text((x + 5, y + 5), text, fill=color)


def draw_ray(draw, p1, p2, color, width=3):
    if p1 is None or p2 is None:
        return
    draw.line([tuple(p1), tuple(p2)], fill=color, width=width)


@torch.no_grad()
def eval_refcoco_iou_metrics(args, model, loader, device, hand_thresh: float = 0.5):
    
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_start_time = time.time()

    all_ious_local = []

    tp = fp = tn = fn = 0
    num_pos = num_neg = 0
    total_samples = 0

    hand_thresh = float(getattr(args, "hand_thresh", hand_thresh))
    eval_with_box_zero_as_neg = bool(getattr(args, "eval_with_box_zero_as_neg", False))

    for batch in loader:
        if not isinstance(batch, (list, tuple)):
            raise TypeError(f"batch must be list/tuple, got {type(batch)}")

        if len(batch) == 4:
            img_data, text_data, target, tgt_mask = batch
            is_positive = None
        elif len(batch) >= 8:
            img_data, text_data, target, tgt_mask, kp_root, kp_tip, kp_valid, is_positive = batch[:8]
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        img_data = img_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)

        if is_positive is not None:
            is_positive = is_positive.to(device).view(-1).long()  # [B]

        # -------------------- model forward --------------------
        outputs = model(img_data.tensors, img_data.mask, text_data)

        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"model outputs must be tuple/list, got {type(outputs)}")
        pred_box = None
        seg_mask = None
        img_cls = None
        text_cls = None
        hand_logit = None
        pred_kp = None
        hand_patch_mask = None

        if len(outputs) >= 7:
            pred_box, seg_mask, img_cls, text_cls, hand_logit, pred_kp, hand_patch_mask = outputs[:7]
        elif len(outputs) == 5:
            pred_box, seg_mask, img_cls, text_cls, hand_logit = outputs
        elif len(outputs) == 4:
            pred_box, seg_mask, img_cls, text_cls = outputs
            hand_logit = None
        # elif len(outputs) >= 5:
        #     pred_box, seg_mask, img_cls, text_cls, hand_logit = outputs[:5]
        else:
            raise ValueError(f"Unexpected model output length at inference: {len(outputs)}")

        B = pred_box.shape[0]

        if is_positive is None:
            _, iou = eval_utils.trans_vg_eval_test_iou(
                pred_box.detach().cpu(), target.detach().cpu()
            )
            all_ious_local.append(iou)
            total_samples += int(B)
            continue

        use_hand_branch = bool(getattr(args, "use_hand_branch", True))
        if use_hand_branch and hand_logit is not None:
            prob = torch.sigmoid(hand_logit.view(-1))   # [B]
            pred_pos = (prob >= hand_thresh).long()     # [B]
        else:
            pred_neg = (pred_box == 0).all(dim=1).long()   # [B], 1 means predicted negative
            pred_pos = 1 - pred_neg                        # [B], 1 means predicted positive


        gt_pos = is_positive.long().view(-1)  # [B]

        # -------------------- confusion matrix --------------------
        num_pos += int((gt_pos == 1).sum().item())
        num_neg += int((gt_pos == 0).sum().item())
        total_samples += int(gt_pos.numel())

        tp += int(((pred_pos == 1) & (gt_pos == 1)).sum().item())
        fp += int(((pred_pos == 1) & (gt_pos == 0)).sum().item())
        tn += int(((pred_pos == 0) & (gt_pos == 0)).sum().item())
        fn += int(((pred_pos == 0) & (gt_pos == 1)).sum().item())

        task_iou = torch.zeros((B,), dtype=torch.float32, device=pred_box.device)

        neg_correct_mask = (gt_pos == 0) & (pred_pos == 0)
        task_iou[neg_correct_mask] = 1.0

        pos_correct_gate_mask = (gt_pos == 1) & (pred_pos == 1)
        if pos_correct_gate_mask.any():
            pb = pred_box[pos_correct_gate_mask]
            gt = target[pos_correct_gate_mask]

            assert gt.min() >= 0 and gt.max() <= 1, \
                f"GT not normalized: min={gt.min().item()}, max={gt.max().item()}"

            _, iou_real = eval_utils.trans_vg_eval_test_iou(
                pb.detach().cpu(), gt.detach().cpu()
            )
            task_iou[pos_correct_gate_mask] = iou_real.to(pred_box.device)

        all_ious_local.append(task_iou.detach().cpu())

    # ------------------------------------------------------------
    # concat IoU
    # ------------------------------------------------------------
    if len(all_ious_local) == 0:
        all_ious = torch.zeros((0,), dtype=torch.float32)
    else:
        all_ious = torch.cat(all_ious_local, dim=0)

    # ------------------------------------------------------------
    # distributed gather
    # ------------------------------------------------------------
    if args.distributed:
        import torch.distributed as dist
        dist.barrier()

        gathered_ious = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_ious, all_ious.cpu())

        gathered_stats = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            gathered_stats,
            (tp, fp, tn, fn, num_pos, num_neg, total_samples)
        )

        if utils.is_main_process():
            all_ious = torch.cat(gathered_ious, dim=0)
            tp = sum(s[0] for s in gathered_stats)
            fp = sum(s[1] for s in gathered_stats)
            tn = sum(s[2] for s in gathered_stats)
            fn = sum(s[3] for s in gathered_stats)
            num_pos = sum(s[4] for s in gathered_stats)
            num_neg = sum(s[5] for s in gathered_stats)
            total_samples = sum(s[6] for s in gathered_stats)
        else:
            return None

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_end_time = time.time()
    total_infer_time = total_end_time - total_start_time

    if not utils.is_main_process():
        return None

    ious = all_ious.tolist()
    n_iou = len(ious)

    if n_iou > 0:
        p03 = sum(i >= 0.3 for i in ious) / n_iou
        p05 = sum(i >= 0.5 for i in ious) / n_iou
        p07 = sum(i >= 0.7 for i in ious) / n_iou
        miou = sum(ious) / n_iou
    else:
        p03 = p05 = p07 = miou = 0.0

    total_cls = tp + fp + tn + fn
    hand_acc = (tp + tn) / total_cls if total_cls > 0 else 0.0
    hand_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    hand_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    hand_f1 = (
        2 * hand_precision * hand_recall / (hand_precision + hand_recall)
        if (hand_precision + hand_recall) > 0 else 0.0
    )

    neg_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    avg_sample_time_ms = (total_infer_time / total_samples * 1000.0) if total_samples > 0 else 0.0
    throughput = (total_samples / total_infer_time) if total_infer_time > 0 else 0.0

    return {
        # unified task-iou metrics
        "num_samples_for_iou": int(n_iou),
        "precision@0.3": float(p03),
        "precision@0.5": float(p05),
        "precision@0.7": float(p07),
        "mIoU": float(miou),

        # cls metrics
        "num_pos": int(num_pos),
        "num_neg": int(num_neg),
        "total_samples": int(total_samples),

        # speed metrics
        "total_inference_time_sec": float(total_infer_time),
        "avg_sample_time_ms": float(avg_sample_time_ms),
        "throughput_samples_per_sec": float(throughput),
    }


def parse_finetuned_dataset_from_ckpt(ckpt_path: str):
    # e.g. .../finetuning_checkpoints/referit/best_checkpoint.pth
    parts = os.path.normpath(ckpt_path).split(os.sep)
    if "finetuning_checkpoints" in parts:
        idx = parts.index("finetuning_checkpoints")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_config = _model_config(args)
    print("model_config =", model_config)

    model = create_model(
        model_config,
        sys_args=args,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    ckpt = torch.load(args.eval_model, map_location="cpu")
    #===================================================================================
    #missing, unexpected = model_without_ddp.load_state_dict(ckpt["model"], strict=False)
    ckpt_state = ckpt["model"]
    model_state = model_without_ddp.state_dict()

    filtered_state = {}
    skipped = []

    for k, v in ckpt_state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped.append((k, v.shape, model_state[k].shape))
        else:
            skipped.append((k, v.shape, None))

    missing, unexpected = model_without_ddp.load_state_dict(filtered_state, strict=False)

    print("\n===== Skipped parameters due to mismatch or missing key =====")
    for item in skipped:
        print(item)

    print("\n===== Missing keys in current model (kept default init) =====")
    for k in missing:
        print(k)

    print("\n===== Unexpected keys from checkpoint =====")
    for k in unexpected:
        print(k)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    dataset_test = build_dataset(args.eval_set, args)

    if args.distributed:
        sampler = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset_test)

    loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,  
        num_workers=args.num_workers,
    )

    start = time.time()
    #metrics = eval_refcoco_iou_metrics(args, model, loader, device)
    if getattr(args, "visualize", False):
        visualize_predictions(
            args=args,
            model=model,
            loader=loader,
            device=device,
            num_vis=getattr(args, "num_vis", 30),
        )
        
    metrics = eval_refcoco_iou_metrics(args, model, loader, device, hand_thresh=getattr(args, "hand_thresh", 0.5))

    if utils.is_main_process():
        finetuned_on = parse_finetuned_dataset_from_ckpt(args.eval_model)
        dataset_name = args.dataset.upper()
        split_name = args.eval_set
        print("=" * 70)
        print(f"Model finetuned on : {finetuned_on}")
        print(f"Evaluated on       : {args.dataset} ({args.eval_set})")
        print("=" * 70)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"compare_{args.dataset}_{args.eval_set}.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        total = time.time() - start
        print("Total time:", str(datetime.timedelta(seconds=int(total))))
        print("\n================= Inference Speed =================")
        print(f"Total inference time  : {metrics['total_inference_time_sec']:.2f} s")
        print(f"Avg sample time       : {metrics['avg_sample_time_ms']:.2f} ms")
        print(f"Throughput            : {metrics['throughput_samples_per_sec']:.2f} samples/s")
        print("===================================================\n")

        with (out / f"compare_{args.dataset}_{args.eval_set}.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        total = time.time() - start
        print("Wall-clock total time:", str(datetime.timedelta(seconds=int(total))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
