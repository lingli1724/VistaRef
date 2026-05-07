# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np

# import utils
# import .beit3_utils as utils
import utils
from .utils import ClipLoss, get_rank, get_world_size, BertCaptioningLoss, mdetr_interpolate

from .modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .mlm_decoder import build_transformer
from torchvision.transforms import Resize
import math
import spacy
from spacy.tokens.token import Token
from fvcore.nn import FlopCountAnalysis


"""
   The OneRef core model code is implemented here.
   Implemented by Linhui Xiao.
      2024-01-10
"""


# SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]  # "xsubj", "npsubj", "subj"
SUBJECTS = ["subj", "xsubj", "npsubj", "nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
# OBJECTS = ["dobj", "dative", "attr", "oprd"]
OBJECTS = ["dobj", "pobj", "dative", "oprd"]
# nlp = spacy.load("en_core_web_sm")  # small, 12MB
nlp = spacy.load("en_core_web_md")  # middle, 42MB


def get_subject_phrase(doc):
    for token in doc:
        if (token.dep_ in SUBJECTS):
            return doc[token.i]
    for token in doc:  # if a noun as the root, just using the root.
        if "ROOT" in token.dep_:
            return doc[token.i]
    return None


def get_predicate_phrase(doc):
    for token in doc:
        if ("ROOT" in token.dep_) and ("VERB" in token.pos_):
            return doc[token.i]
    for token in doc:  # if a noun as the root, just using the root.
        if "VERB" in token.pos_:
            return doc[token.i]
    return None


def get_object_phrase(doc):
    for token in doc:
        if (token.dep_ in OBJECTS):
            return doc[token.i]
    return None


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


class TwoLayerMLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features,
            out_features,
            norm_layer,
            norm_input=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BEiT3ForGrounding(BEiT3Wrapper):
    def __init__(self, sys_args, args, **kwargs):
        super(BEiT3ForGrounding, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.use_hand_branch = sys_args.use_hand_branch
        self.use_kp_branch = sys_args.use_kp_branch
        self.use_ocal_module = sys_args.use_ocal_module
        if self.use_ocal_module and (not self.use_kp_branch):
            raise ValueError("use_ocal_module=True requires use_kp_branch=True")

        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.criterion = ClipLoss(rank=get_rank(), world_size=get_world_size())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if sys_args.frozen_backbone:
            for parameter in self.beit3.parameters():
                parameter.requires_grad_(False)

        self.img_resize = Resize([int(args.img_size / args.patch_size), int(args.img_size / args.patch_size)])  # 24 * 24 = 576
        self.num_visu_token = int((args.img_size / args.patch_size) ** 2)  # 384/16=24*24=576
        self.num_text_token = sys_args.max_query_len
        if self.use_ocal_module:
            bbox_in_dim = embed_dim * 3
        else:
            bbox_in_dim = embed_dim

        self.bbox_embed = MLP(bbox_in_dim, embed_dim, 4, 3)
        self.visu_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        if self.use_kp_branch:
            self.kp_head = MLP(embed_dim, embed_dim, 4, 3)
        else:
            self.kp_head = None
        #self.hand_cls_head = nn.Linear(embed_dim, 1)
        if self.use_hand_branch:
            self.hand_cls_head = nn.Linear(embed_dim, 1)
        else:
            self.hand_cls_head = None
        self.ray_feat_dim = 7
        if self.use_ocal_module:
            self.ray_encoder = MLP(self.ray_feat_dim, embed_dim, embed_dim, 3)
            self.ray_fusion_gate = nn.Parameter(torch.tensor(-2.0))
            self.hand_query_fusion = MLP(embed_dim * 2, embed_dim, embed_dim, 3)
            self.hand_object_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                batch_first=True
            )
            self.pseudo_hand_long_ratio = 2.2
            self.pseudo_hand_short_ratio = 1.0
            self.pseudo_hand_center_ratio = 0.3
            self.pseudo_hand_min_size = 2.0 / int(args.img_size / args.patch_size)
        else:
            self.ray_encoder = None
            self.hand_query_fusion = None
            self.hand_object_attn = None
            self.ray_fusion_gate = None

        if self.kp_head is not None:
            self.kp_head.apply(self._init_weights)
        if self.hand_cls_head is not None:
            self.hand_cls_head.apply(self._init_weights)
        if self.hand_query_fusion is not None:
            self.hand_query_fusion.apply(self._init_weights)

        kp_trainable = any(p.requires_grad for p in self.kp_head.parameters()) \
            if self.kp_head is not None else False

        hand_trainable = any(p.requires_grad for p in self.hand_cls_head.parameters()) \
            if self.hand_cls_head is not None else False

        bbox_trainable = any(p.requires_grad for p in self.bbox_embed.parameters())

        query_trainable = any(p.requires_grad for p in self.hand_query_fusion.parameters()) \
            if self.hand_query_fusion is not None else False

        attn_trainable = any(p.requires_grad for p in self.hand_object_attn.parameters()) \
            if self.hand_object_attn is not None else False

        print(f"[ModelInit] bbox_embed trainable: {bbox_trainable}")
        print(f"[ModelInit] kp_head trainable: {kp_trainable}")
        print(f"[ModelInit] hand_cls_head trainable: {hand_trainable}")
        print(f"[ModelInit] hand_query_fusion trainable: {query_trainable}")
        print(f"[ModelInit] hand_object_attn trainable: {attn_trainable}")

        head_param_1 = sum(p.numel() for p in self.bbox_embed.parameters() if p.requires_grad)
        head_param_2 = sum(p.numel() for p in self.visu_proj.parameters() if p.requires_grad)
        print('number of head params: ', head_param_1 + head_param_2)

        head_param_3 = sum(p.numel() for p in self.kp_head.parameters() if p.requires_grad) \
            if self.kp_head is not None else 0

        head_param_4 = sum(p.numel() for p in self.hand_cls_head.parameters() if p.requires_grad) \
            if self.hand_cls_head is not None else 0

        head_param_5 = sum(p.numel() for p in self.hand_query_fusion.parameters() if p.requires_grad) \
            if self.hand_query_fusion is not None else 0

        head_param_6 = sum(p.numel() for p in self.hand_object_attn.parameters() if p.requires_grad) \
            if self.hand_object_attn is not None else 0

        print(
            'number of head params (bbox+visu_proj+kp+hand_cls): ',
            head_param_1 + head_param_2 + head_param_3 + head_param_4 + head_param_5 + head_param_6
        )

        # Initialize the tokenizer based on the passed-in text tokenizer information.
        self.tokenizer = get_sentencepiece_model_for_beit3(sys_args)
        self.num_max_bpe_tokens = sys_args.max_query_len  # self.num_max_bpe_tokens = num_max_bpe_tokens, default 64
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.enable_ref_mlm = sys_args.enable_ref_mlm
        self.text_mask_prob = sys_args.text_mask_prob  # by args.captioning_mask_prob，usually 0.7
        self.mask_token_id = self.tokenizer.mask_token_id
        self.language_vocab_size = self.tokenizer.vocab_size

        self.use_contrastive_loss = sys_args.use_contrastive_loss
        self.use_mask_loss = sys_args.use_mask_loss
        if self.use_mask_loss:
            deconv_hidden_dim = args.encoder_embed_dim
            self.seg_conv1 = nn.ConvTranspose2d(in_channels=deconv_hidden_dim, out_channels=deconv_hidden_dim,
                                                kernel_size=(2, 2), stride=(2, 2), padding=(0, 0),
                                                output_padding=(0, 0), bias=False)  # bias=False
            self.bn1 = nn.BatchNorm2d(deconv_hidden_dim)
            self.seg_conv2 = nn.ConvTranspose2d(in_channels=deconv_hidden_dim, out_channels=deconv_hidden_dim,
                                                kernel_size=(2, 2), stride=(2, 2), padding=(0, 0),
                                                output_padding=(0, 0), bias=False)  # bias=False
            self.bn2 = nn.BatchNorm2d(deconv_hidden_dim)
            self.seg_conv3 = nn.ConvTranspose2d(in_channels=deconv_hidden_dim, out_channels=deconv_hidden_dim,
                                                kernel_size=(2, 2), stride=(2, 2), padding=(0, 0),
                                                output_padding=(0, 0), bias=False)  # bias=False
            self.bn3 = nn.BatchNorm2d(deconv_hidden_dim)
            self.relu = nn.ReLU(inplace=True)

        self.enable_ref_mim = sys_args.enable_ref_mim
        self.mim_mid_layer = sys_args.mim_mid_layer
        self.return_all_hiddens = True if self.mim_mid_layer else False
        self.layer_norm = nn.LayerNorm(embed_dim) if self.mim_mid_layer else None
        if self.enable_ref_mim:
            self.shared_lm_head = True  # Default sharing the mask head
            self.mim_head = nn.Linear(embed_dim, sys_args.codebook_size)
            self.mim_vts_head = MLP(embed_dim, embed_dim, 4, 3)  # visual target-relation score

        if self.enable_ref_mlm:
            self.mlm_head = nn.Linear(embed_dim, args.vocab_size)
            self.mlm_head.apply(self._init_weights)
            self.mlm_loss = BertCaptioningLoss(sys_args.label_smoothing, sys_args.drop_worst_ratio, sys_args.drop_worst_after)
            self.mlm_sts_head = MLP(embed_dim, embed_dim, 1, 3)  # semantic target-relation score
    
    def _extract_pseudo_hand_feat(self, patch_feat, pred_root, pred_tip):
        device = patch_feat.device
        B, N, H = patch_feat.shape
        grid_size = int(math.sqrt(N))
        assert grid_size * grid_size == N, "patch number must form a square grid"

        ys = (torch.arange(grid_size, device=device, dtype=patch_feat.dtype) + 0.5) / grid_size
        xs = (torch.arange(grid_size, device=device, dtype=patch_feat.dtype) + 0.5) / grid_size
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [G, G]
        grid_centers = torch.stack([grid_x, grid_y], dim=-1).reshape(1, N, 2).repeat(B, 1, 1)  # [B, N, 2]

        delta = pred_tip - pred_root                                # [B, 2]
        L = torch.norm(delta, dim=-1, keepdim=True)                 # [B, 1]
        unit = delta / (L + 1e-6)                                   # [B, 2]

        center = pred_root + self.pseudo_hand_center_ratio * delta  # [B, 2]

        long_side = self.pseudo_hand_long_ratio * L                 # [B, 1]
        short_side = self.pseudo_hand_short_ratio * L               # [B, 1]

        long_side = torch.clamp(long_side, min=self.pseudo_hand_min_size)
        short_side = torch.clamp(short_side, min=self.pseudo_hand_min_size)

        ux = unit[:, 0:1]
        uy = unit[:, 1:2]
        vx = -uy
        vy = ux

        half_x = torch.abs(ux) * (long_side * 0.5) + torch.abs(vx) * (short_side * 0.5)  # [B,1]
        half_y = torch.abs(uy) * (long_side * 0.5) + torch.abs(vy) * (short_side * 0.5)  # [B,1]

        x1 = torch.clamp(center[:, 0:1] - half_x, min=0.0, max=1.0)
        y1 = torch.clamp(center[:, 1:2] - half_y, min=0.0, max=1.0)
        x2 = torch.clamp(center[:, 0:1] + half_x, min=0.0, max=1.0)
        y2 = torch.clamp(center[:, 1:2] + half_y, min=0.0, max=1.0)

        gx = grid_centers[:, :, 0]   # [B, N]
        gy = grid_centers[:, :, 1]   # [B, N]

        hand_patch_mask = (gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)  # [B, N]

        empty_mask = (hand_patch_mask.sum(dim=-1) == 0)  # [B]
        if empty_mask.any():
            dist2 = (grid_centers[:, :, 0] - center[:, 0:1]) ** 2 + (grid_centers[:, :, 1] - center[:, 1:2]) ** 2
            nearest_idx = dist2.argmin(dim=-1)  # [B]
            fallback_mask = torch.zeros_like(hand_patch_mask)
            fallback_mask[torch.arange(B, device=device), nearest_idx] = True
            hand_patch_mask = torch.where(empty_mask.unsqueeze(-1), fallback_mask, hand_patch_mask)

        hand_patch_mask_float = hand_patch_mask.float().unsqueeze(-1)  # [B, N, 1]
        hand_feat = (patch_feat * hand_patch_mask_float).sum(dim=1) / (hand_patch_mask_float.sum(dim=1) + 1e-6)  # [B, H]

        return hand_feat, hand_patch_mask


    def mlm_criterion(self, masked_feats):
        pass

    """ The purpose of this function is to add some perturbations to the mask to increase its diversity. """
    def _get_mask_token(self, token):
        p = random.random()
        if p < 0.8:
            return self.mask_token_id  # The selected position has a 0.8 probability of being assigned a mask token
        elif p < 0.9:
            return token  # There is a 0.1 probability of no change.
        else:
            return random.randint(3, self.language_vocab_size - 1)  # There is a 0.1 probability of random masking.

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens  # Set the maximum length

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]  # The ending token values are 0 and 2 .
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_token = tokens + [self.pad_token_id] * (max_len - num_tokens)
        return language_token, padding_mask, num_tokens

    def _masking_on_text_tokens_v1(self, tokens, num_tokens, mask_prob):
        bool_masked_pos = [0] * len(tokens)  # First, initialize a position vector of the mask with all zeros.
        # The number of tokens that obtain the mask, with the maximum masked quantity being the length of the original
        # tokens. The reason for adding the min operation is to prevent the input mask_prob from being greater than or
        # equal to 1.0.
        # to_mask = min(int(num_tokens) * mask_prob + 0.5), num_tokens - 1)  # The 0th position is not masked.
        to_mask = min(int((num_tokens - 1) * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)  # The minimum number of masks is 1.
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)  # Randomly select a position from the range [1, num_token - 1].
            if bool_masked_pos[i] == 0:  # If there is no mask at this position
                bool_masked_pos[i] = 1  # Then set the mask position.
                tokens[i] = self._get_mask_token(tokens[i])  # Mask token at this position, with the mask token is 64001
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def _masking_on_text_tokens_v2(self, tokens, num_tokens, mask_prob, sbj_idx, obj_idx):
        bool_masked_pos = [0] * len(tokens)
        to_mask = min(int((num_tokens - 1) * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)  # The minimum number of masks is 1.

        num_masked_tokens = 0

        if num_masked_tokens < to_mask and sbj_idx is not None:
            if bool_masked_pos[sbj_idx + 1] == 0:
                bool_masked_pos[sbj_idx + 1] = 1
                tokens[sbj_idx + 1] = self._get_mask_token(tokens[sbj_idx + 1])
                num_masked_tokens += 1

        if num_masked_tokens < to_mask and obj_idx is not None:
            if bool_masked_pos[obj_idx + 1] == 0:
                bool_masked_pos[obj_idx + 1] = 1
                tokens[obj_idx + 1] = self._get_mask_token(tokens[obj_idx + 1])
                num_masked_tokens += 1

        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                tokens[i] = self._get_mask_token(tokens[i])
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def _get_text_token_and_padding_mask(self, text_batch, device):
        language_tokens, padding_masks = [], []
        for text in text_batch:
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # At this point, in the mask, the positions with content are marked as 0, and those without content
            # are marked as 1.
            language_token, padding_mask, num_tokens = self._get_text_segment(token_ids)
            language_tokens.append(language_token)
            padding_masks.append(padding_mask)
        return torch.tensor(language_tokens).to(device), torch.tensor(padding_masks).to(device)

    def _find_index(self, string_list, target_string):
        try:
            index = string_list.index(target_string)
            return index
        except ValueError:
            return None

    def _get_text_token_and_masking_modeling(self, text_batch, device):
        language_tokens, padding_masks = [], []
        masked_tokens, language_masked_poses = [], []
        for text in text_batch:
            doc = nlp(text)
            subject = get_subject_phrase(doc)
            object = get_object_phrase(doc)
            tokens = self.tokenizer.tokenize(text)
            sbj_idx = self._find_index(list(tokens), '▁' + subject.text) if subject else None
            obj_idx = self._find_index(list(tokens), '▁' + object.text) if object else None
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # token_ids is a list of integers.
            language_token, padding_mask, num_token = self._get_text_segment(token_ids)
            masked_token = language_token[:]
            # mask sub and obj
            masked_token, language_masked_pos = \
                self._masking_on_text_tokens_v2(masked_token, num_token, self.text_mask_prob, sbj_idx, obj_idx)

            language_tokens.append(language_token)
            padding_masks.append(padding_mask)
            masked_tokens.append(masked_token)
            language_masked_poses.append(language_masked_pos)

        return torch.tensor(language_tokens).to(device), torch.tensor(padding_masks).to(device), \
               torch.tensor(masked_tokens).to(device), torch.tensor(language_masked_poses).to(device)

    def forward(self, image=None, img_mask=None, text=None, text_len=None, global_step=None, mim_masked_pos=None,
            use_plain_text=True, obj_mask=None, enable_ref_mim=None, enable_ref_mlm=None, training=False, **kwargs):
        assert image is not None and text is not None
        debug_forward = kwargs.get("debug_forward", False)

        # Fallback to model config if caller does not explicitly pass these flags
        if enable_ref_mlm is None:
            enable_ref_mlm = self.enable_ref_mlm
        if enable_ref_mim is None:
            enable_ref_mim = self.enable_ref_mim

        masked_tokens, language_masked_pos = None, None
        if use_plain_text:  # default
            if training and enable_ref_mlm:
                text_tokens, padding_mask, masked_tokens, language_masked_pos = \
                    self._get_text_token_and_masking_modeling(text, image.device)
            else:
                text_tokens, padding_mask = self._get_text_token_and_padding_mask(text, image.device)
        else:
            text_tokens, padding_mask = text.tensors, text.mask

        # encode image and text
        batch_size = image.shape[0]
        image_len = self.beit3.vision_embed.num_position_embeddings()  # include cls token
        text_len = text_len if text_len is not None else text_tokens.size(1)

        if training and enable_ref_mlm:
            outputs = self.beit3(
                textual_tokens=masked_tokens,
                visual_tokens=image,
                text_padding_position=padding_mask,
                vision_masked_position=mim_masked_pos,
                return_all_hiddens=self.return_all_hiddens
            )
        else:
            outputs = self.beit3(
                textual_tokens=text_tokens,
                visual_tokens=image,
                text_padding_position=padding_mask,
                vision_masked_position=mim_masked_pos,
                return_all_hiddens=self.return_all_hiddens
            )

        vision_feat = outputs["encoder_out"][:, 0:image_len]   # [B, Lv, H]
        language_feat = outputs["encoder_out"][:, image_len:]  # [B, Lt, H]

        # contrastive features
        vision_contrastive = F.normalize(self.vision_head(vision_feat), dim=-1)
        vision_cls = vision_contrastive[:, 0, :].contiguous()
        vision_norm_token = vision_contrastive[:, 1:, :].contiguous()

        language_contrastive = F.normalize(self.language_head(language_feat), dim=-1)
        language_cls = language_contrastive[:, 0, :].contiguous()
        language_norm_token = language_contrastive[:, :, :].contiguous()

        # visual token cosine similarity
        visu_token_dot_product_matrix = torch.mul(
            language_cls.unsqueeze(1).repeat(1, image_len - 1, 1),
            vision_norm_token
        )
        visu_token_similarity = visu_token_dot_product_matrix.sum(axis=-1, keepdim=False)

        text_token_dot_product_matrix = torch.mul(
            vision_cls.unsqueeze(1).repeat(1, text_len, 1),
            language_norm_token
        )

        # visual grounding pooled feature
        visu_src = self.visu_proj(vision_feat)  # [B, L, H]
        vg_hs = torch.mul(
            visu_token_similarity.softmax(dim=-1).unsqueeze(-1).repeat(1, 1, visu_src.shape[-1]),
            visu_src[:, 1:, :]
        )
        vg_hs = vg_hs.sum(axis=1, keepdim=False)  # [B, H]
        B, H = vg_hs.shape
        device = vg_hs.device
        dtype = vg_hs.dtype

        # default outputs
        hand_logit = None
        pred_kp = None
        hand_patch_mask = None
        attn_weights = None

        if self.use_hand_branch:
            hand_logit = self.hand_cls_head(vg_hs).squeeze(-1) # [B]
        if self.use_kp_branch:
            pred_kp = self.kp_head(vg_hs).sigmoid()            # [B, 4]
            pred_root = pred_kp[:, 0:2]
            pred_tip = pred_kp[:, 2:4]
            ray_delta = pred_tip - pred_root
            ray_len = torch.norm(ray_delta, dim=-1, keepdim=True)
            ray_unit = ray_delta / (ray_len + 1e-6)

            ray_geom = torch.cat([pred_root, pred_tip, ray_unit, ray_len], dim=-1)

            if self.use_ocal_module:
                ray_embed = self.ray_encoder(ray_geom)

                patch_feat = visu_src[:, 1:, :]  # [B, N, H]

                hand_feat, hand_patch_mask = self._extract_pseudo_hand_feat(
                    patch_feat=patch_feat,
                    pred_root=pred_root,
                    pred_tip=pred_tip
                )  # hand_feat: [B, H], hand_patch_mask: [B, N]

                hand_ray_query = self.hand_query_fusion(
                    torch.cat([hand_feat, ray_embed], dim=-1)
                )  # [B, H]

                attn_out, attn_weights = self.hand_object_attn(
                    query=hand_ray_query.unsqueeze(1),
                    key=patch_feat,
                    value=patch_feat,
                    need_weights=True
                )
                target_feat = attn_out.squeeze(1)      # [B, H]
                attn_weights = attn_weights.squeeze(1) # [B, N]
            else:
                hand_patch_mask = None
                attn_weights = None
                target_feat = torch.zeros(B, H, device=device, dtype=dtype)

        else:
            # kp branch disabled => do not use geometry path
            pred_kp = None
            hand_patch_mask = None
            ray_embed = torch.zeros(B, H, device=device, dtype=dtype)
            target_feat = torch.zeros(B, H, device=device, dtype=dtype)
            attn_weights = None

        ray_gate = torch.tensor(6.0, device=vg_hs.device)
        #fused_feat = torch.cat([vg_hs, ray_gate * ray_embed], dim=-1)
        #fused_feat = torch.cat([vg_hs, ray_embed], dim=-1)
        if self.use_ocal_module:
            fused_feat = torch.cat([vg_hs, ray_gate * ray_embed, target_feat], dim=-1)  # [B, 3H]
        else:
            fused_feat = vg_hs

        pred_box = self.bbox_embed(fused_feat).sigmoid()

        vision_mask_feat = vision_feat[:, 1:]  # [B, Lv-1, H]
        patch_num = int(math.sqrt(vision_mask_feat.shape[1]))
        channel = vision_mask_feat.shape[2]
        assert patch_num * patch_num == vision_mask_feat.shape[1]

        seg_mask = torch.tensor([], device=image.device)
        if self.use_mask_loss:
            seg_features = vision_mask_feat.permute(0, 2, 1).reshape(batch_size, channel, patch_num, patch_num)
            seg_features = self.bn3(
                self.seg_conv3(
                    self.relu(
                        self.bn2(
                            self.seg_conv2(
                                self.relu(
                                    self.bn1(self.seg_conv1(seg_features))
                                )
                            )
                        )
                    )
                )
            )
            seg_features = F.normalize(seg_features.permute(0, 2, 3, 1), dim=-1)
            seg_mask = torch.mul(
                language_cls.reshape(batch_size, 1, 1, language_cls.shape[-1]).repeat(
                    1, seg_features.shape[1], seg_features.shape[2], 1
                ),
                seg_features
            )
            seg_mask = seg_mask.sum(axis=-1, keepdim=False).unsqueeze(1)  # [B,1,H,W]

        if training:
            mlm_loss, mlm_acc, mlm_sts_pred = None, None, None
            contrast_loss, mim_pred, mim_vts_pred = None, None, None

            if enable_ref_mlm:
                use_vanilla_text_token = True
                if use_vanilla_text_token:
                    masked_feats = language_feat[language_masked_pos.bool()]
                else:
                    masked_feats = text_token_dot_product_matrix[language_masked_pos.bool()]

                mlm_logits = self.mlm_head(masked_feats)
                masked_labels = text_tokens[language_masked_pos.bool()]
                score = torch.max(mlm_logits, -1)[1].data == masked_labels
                mlm_acc = torch.sum(score.float()) / torch.sum(language_masked_pos)
                mlm_loss = self.mlm_loss(mlm_logits, masked_labels, global_step)
                mlm_sts_pred = self.mlm_sts_head(
                    text_token_dot_product_matrix[:, 1:text_len - 1, :].contiguous()
                ).squeeze().softmax(dim=-1)

            if enable_ref_mim:
                if self.mim_mid_layer:
                    mim_pred = [
                        self.mim_head(vision_feat[:, 1:][mim_masked_pos]),
                        self.mim_head(self.layer_norm(outputs['encoder_states'][self.mim_mid_layer][:, 1:][mim_masked_pos]))
                    ]
                else:
                    mim_pred = self.mim_head(F.normalize(visu_token_dot_product_matrix, dim=-1)[mim_masked_pos])
                    mim_vts_pred = self.mim_vts_head(visu_token_dot_product_matrix)

            if self.use_contrastive_loss:
                contrast_loss, logits_per_image, logits_per_text = \
                    self.criterion(vision_cls, language_cls, self.logit_scale.exp())

            # Return 11 outputs in training mode
            return (
                pred_box,
                contrast_loss,
                visu_token_similarity,
                seg_mask,
                mlm_loss,
                mlm_acc,
                mlm_sts_pred,
                mim_pred,
                mim_vts_pred,
                pred_kp,
                hand_logit,
            )

        else:
            # inference path: keep backward compatibility for existing eval code
            return pred_box, seg_mask, vision_cls, language_cls, hand_logit, pred_kp, hand_patch_mask


@register_model
def beit3_base_patch16_384_grounding(sys_args, pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)  # Return the basic model configuration information.
    model = BEiT3ForGrounding(sys_args, args, **kwargs)
    return model


@register_model
def beit3_base_patch16_480_grounding(sys_args, pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)  # Return the basic model configuration information.
    model = BEiT3ForGrounding(sys_args, args, **kwargs)
    return model


@register_model
def beit3_large_patch16_384_grounding(sys_args, pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)  # Return the basic model configuration information.
    model = BEiT3ForGrounding(sys_args, args, **kwargs)
    return model


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

