# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data

sys.path.append('.')


from PIL import Image
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus
from pycocotools import mask as coco_mask

# from CLIP-VG.models.clip import *
# import clip


def convert_coco_poly_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        # polygons 可能是：
        # 1) 单个 polygon: [x1,y1,...]
        # 2) 多个 polygon: [[x1,y1,...],[...],...]
        if len(polygons) > 0 and isinstance(polygons[0], (int, float)):
            polygons = [polygons]  # 关键：wrap 成 list-of-polygons

        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        # If the mask is empty, it indicates that there is no target. Return a mask with a value of 0 directly.
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


class TransVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test', 'train_pseudo')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB', 'train_pseudo'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB', 'train_pseudo'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val', 'train_pseudo'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test', 'train_pseudo'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test', 'train_pseudo')
        },
        'mixup': {
            'splits': ('train', 'val', 'test', 'train_pseudo')
        },
        'egopoint': {'splits': ('test', 'val', 'train')}   
    }

    def __init__(self, args, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, prompt_template=None, lstm=False,
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.prompt_template = prompt_template
        self.transform = transform
        self.testmode = testmode
        self.split = split
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx = return_idx
        self.enable_seg_mask = args.enable_seg_mask

        """" add by xiaolinhui """
        # Initialize the tokenizer based on the passed-in text tokenizer information.
        self.tokenizer = get_sentencepiece_model_for_beit3(args)
        self.num_max_bpe_tokens = max_query_len  # self.num_max_bpe_tokens = num_max_bpe_tokens default equal 64
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        assert self.transform is not None

        if split in ['train', 'train_pseudo']:
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            # from flickr30k_images to flickr30k-images
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        elif self.dataset == 'egopoint':
            self.dataset_root = self.data_root
            self.im_dir = osp.join(self.dataset_root, 'img', split)
        else:  # refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('The dataset {} is not found!'.format(osp.join(self.split_root, self.dataset)))
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            data = torch.load(imgset_path)

            if len(self.images) == 0:   
                print("==== EgoPoint pth sanity check ====")
                print("type(data):", type(data), "len(data):", len(data))
                x = data[0]
                print("type(x):", type(x))
                if isinstance(x, (list, tuple)):
                    print("len(x):", len(x))
                    print("x[0]:", type(x[0]), x[0])
                    print("x[1]:", type(x[1]), x[1])
                    print("x[2]:", type(x[2]), x[2])
                    print("x[3]:", type(x[3]), x[3])
                    print("x[4]:", type(x[4]), x[4])
                else:
                    print("x keys:", list(x.keys()))
                print("===================================")

            self.images += data
            #self.images += torch.load(imgset_path)

        # if self.prompt_template:
        #     self.images = self.prompt(self.images)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':  # flickr
            img_file, bbox, phrase = self.images[idx]
            img_size = None
            obj_mask = None
            image_size = []
            bbox_xywh = bbox.copy()

            kp_root = [0.0, 0.0]
            kp_tip = [0.0, 0.0]
            kp_valid = 0
            is_positive = 1

        else:
            item = self.images[idx]
            if len(item) == 5:
                img_file, img_size, bbox, phrase, obj_mask = item
                kp_root = [0.0, 0.0]
                kp_tip = [0.0, 0.0]
                kp_valid = 0
                is_positive = 1   
            elif len(item) >= 9:
                img_file, img_size, bbox, phrase, obj_mask, kp_root, kp_tip, kp_valid, is_positive = item[:9]
            else:
                raise ValueError(f"bad self.images[{idx}] len={len(item)} item={item}")

            assert isinstance(img_size, dict) and "height" in img_size and "width" in img_size, \
                f"bad img_size: type={type(img_size)}, value={img_size}"

            bbox_xywh = bbox.copy()
            image_size = [img_size["height"], img_size["width"]]

        bbox_ori = torch.tensor(bbox.copy(), dtype=torch.float32)

        """
        For refcoco-style datasets:
        xywh -> x1y1x2y2
        referit/flickr keep x1y1x2y2, while bbox_xywh is prepared for mask generation
        """
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)
            bbox_xywh[2], bbox_xywh[3] = bbox_xywh[2] - bbox_xywh[0], bbox_xywh[3] - bbox_xywh[1]

        if self.dataset == "mixup":
            if img_file.split("_")[0] == "COCO":
                dataset = "coco"
                im_dir = osp.join(self.data_root, 'other', 'images', 'mscoco', 'images', 'train2014')
            else:
                if img_size == "flickr":
                    dataset = "flickr"
                    im_dir = osp.join(self.data_root, 'Flickr30k', 'flickr30k-images')
                elif img_size == "referit":
                    dataset = "referit"
                    im_dir = osp.join(self.data_root, 'referit', 'images')
                else:
                    print("img_file：", img_file, 'img_size: ', img_size, "bbox: ", bbox, "phrases: ", phrase)
                    raise ValueError('Can not find image dir')
        else:
            dataset = self.dataset
            im_dir = self.im_dir

        img_path = osp.join(im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        if dataset == 'referit' or dataset == 'flickr':
            image_size = [img.height, img.width]

        bbox = torch.tensor(bbox, dtype=torch.float32)

        if dataset in ["unc", "unc+", "gref", "gref_umd", "coco"]:
            h, w = image_size[0], image_size[1]
            if self.enable_seg_mask:
                bool_obj_mask = convert_coco_poly_mask([obj_mask], h, w)
            else:
                obj_mask = [bbox_xywh]  # xywh
                bool_obj_mask = convert_coco_poly_mask(np.array([obj_mask]), h, w)

        else:
            h, w = image_size[0], image_size[1]
            if dataset == "egopoint":
                seg = obj_mask

                if seg is None:
                    seg = []
                if len(seg) > 0 and isinstance(seg[0], (int, float)):
                    seg = [seg]

                bool_obj_mask = convert_coco_poly_mask(seg, h, w)
            else:
                obj_mask = [list(map(float, bbox_xywh))]
                bool_obj_mask = convert_coco_poly_mask(np.array([obj_mask]), h, w)

        if torch.is_tensor(bool_obj_mask) and bool_obj_mask.dim() == 3:
            bool_obj_mask = bool_obj_mask.any(dim=0)
        bool_obj_mask = bool_obj_mask.to(torch.bool)

        kp_root = torch.tensor(kp_root, dtype=torch.float32)
        kp_tip = torch.tensor(kp_tip, dtype=torch.float32)
        kp_valid = torch.tensor(int(kp_valid), dtype=torch.int64)
        is_positive = torch.tensor(int(is_positive), dtype=torch.int64)

        return img_file, img, phrase, bbox, bbox_ori, bool_obj_mask, kp_root, kp_tip, kp_valid, is_positive

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def prompt(self, sample_list):
        n = len(sample_list)
        new_sample_list = []

        for i in range(n):
            if self.dataset == 'flickr':
                tmp_sample = (sample_list[i][0], sample_list[i][1], self.prompt_template.replace('{pseudo_query}', sample_list[i][2]))
            else:
                tmp_sample = (sample_list[i][0], sample_list[i][1], sample_list[i][2],
                              self.prompt_template.replace('{pseudo_query}', sample_list[i][3]), sample_list[i][4])
            new_sample_list.append(tmp_sample)
        return new_sample_list

    def __len__(self):
        return len(self.images)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens 

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def __getitem__(self, idx):

        img_file, img, phrase, bbox, bbox_ori, obj_mask, kp_root, kp_tip, kp_valid, is_positive = self.pull_item(idx)

        if not torch.is_tensor(bbox):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox = bbox.float()

        if not torch.is_tensor(bbox_ori):
            bbox_ori = torch.tensor(bbox_ori, dtype=torch.float32)
        else:
            bbox_ori = bbox_ori.float()

        if kp_root is None:
            kp_root = torch.zeros(2, dtype=torch.float32)
        elif not torch.is_tensor(kp_root):
            kp_root = torch.tensor(kp_root, dtype=torch.float32)
        else:
            kp_root = kp_root.float()

        if kp_tip is None:
            kp_tip = torch.zeros(2, dtype=torch.float32)
        elif not torch.is_tensor(kp_tip):
            kp_tip = torch.tensor(kp_tip, dtype=torch.float32)
        else:
            kp_tip = kp_tip.float()

        if kp_valid is None:
            kp_valid = torch.tensor(0, dtype=torch.long)
        elif not torch.is_tensor(kp_valid):
            kp_valid = torch.tensor(kp_valid, dtype=torch.long)
        else:
            kp_valid = kp_valid.long()

        if is_positive is None:
            is_positive = torch.tensor(0, dtype=torch.long)
        elif not torch.is_tensor(is_positive):
            is_positive = torch.tensor(is_positive, dtype=torch.long)
        else:
            is_positive = is_positive.long()

        if not torch.is_tensor(obj_mask):
            obj_mask = torch.tensor(obj_mask)
        obj_mask = obj_mask.to(torch.bool)

        if obj_mask.dim() == 2:
            obj_mask = obj_mask.unsqueeze(0)

        input_dict = {
            'img': img,              # PIL
            'box': bbox.clone(),     # x1y1x2y2
            'text': phrase,
            'obj_mask': obj_mask,    # [1,H,W] bool
            'kp_root': kp_root.clone(),
            'kp_tip': kp_tip.clone(),
            'kp_valid': kp_valid.clone(),
            'is_positive': is_positive.clone(),
        }

        if hasattr(self, "transform") and self.transform is not None:
            input_dict = self.transform(input_dict)

        # transform 输出
        img = input_dict['img']                  # Tensor [3,S,S]
        img_mask = input_dict['mask']            # Tensor [S,S]
        bbox = input_dict['box']                 # NormalizeAndPad 后应为 xywh / normalized
        obj_mask = input_dict['obj_mask']        # Tensor [1,S,S]
        phrase = input_dict['text']
        kp_root = input_dict['kp_root']
        kp_tip = input_dict['kp_tip']
        kp_valid = input_dict['kp_valid']
        is_positive = input_dict['is_positive']

        obj_mask = obj_mask.to(torch.bool)

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            tok = self.tokenizer(
                phrase,
                padding="max_length",
                truncation=True,
                max_length=getattr(self, "max_query_len", 40)
            )

            input_ids = tok["input_ids"]
            attn_mask = tok["attention_mask"]

            if isinstance(input_ids[0], (list, tuple)):
                input_ids = input_ids[0]
            if isinstance(attn_mask[0], (list, tuple)):
                attn_mask = attn_mask[0]

            word_id = torch.tensor(input_ids, dtype=torch.long)          # [L]
            word_mask = torch.tensor(attn_mask, dtype=torch.bool)        # [L]
        else:
            L = getattr(self, "max_query_len", 40)
            word_id = torch.zeros(L, dtype=torch.long)
            word_mask = torch.zeros(L, dtype=torch.bool)

        return (
            img,          # 0
            img_mask,     # 1
            word_id,      # 2
            word_mask,    # 3
            bbox,         # 4
            phrase,       # 5
            bbox_ori,     # 6
            obj_mask,     # 7
            kp_root,      # 8
            kp_tip,       # 9
            kp_valid,     # 10
            is_positive,  # 11
        )

        def check_xyxy(bb, W, H):
            x1, y1, x2, y2 = map(float, bb)
            ok = (0 <= x1 < x2 <= W) and (0 <= y1 < y2 <= H)
            return ok, (x1, y1, x2, y2)

        def check_xywh(bb, W, H):
            x, y, w, h = map(float, bb)
            ok = (0 <= x <= W) and (0 <= y <= H) and (w > 0) and (h > 0) and ((x + w) <= W) and ((y + h) <= H)
            return ok, (x, y, w, h)

        xyxy_ok, xyxy_v = check_xyxy(b, W, H)
        xywh_ok, xywh_v = check_xywh(b, W, H)

        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase, 'obj_mask': obj_mask}
        input_dict = self.transform(input_dict)

        img = input_dict['img']
        img_mask = input_dict['mask']  # The mask has been processed in the transform stage.
        bbox = input_dict['box']
        phrase = input_dict['text']
        obj_mask = input_dict['obj_mask']

        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

        # HiVG version, no encoding performed.
        text = []
        text_mask = []
        # Beit3 version. Below is the tokenizer code for Beit-3.
        # tokens = self.tokenizer.tokenize(phrase)
        # token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # language_tokens, padding_mask, _ = self._get_text_segment(token_ids)  # 此时的 mask, 有内容的为 0，没内容的为 1
        # text = language_tokens
        # text_mask = padding_mask

        # text_mask = torch.tensor(text_token > 0).int()[0].numpy().tolist()
        # print("\ntext_token: ", text_token)
        # print('\ntext: ', text)
        """ # old code
        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            # print(img.shape)
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
        """

        if self.testmode:  # default False
            return img, np.array(text, dtype=int), np.array(text_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:  # Need avoid 7 variables.
            return img, np.array(img_mask), np.array(text, dtype=int), np.array(text_mask, dtype=int), np.array(bbox, dtype=np.float32), img_file, phrase, bbox_ori, np.array(obj_mask, dtype=int)


    def getitem_for_origin_transvg(self, idx):

        img_file, img, phrase, bbox, bbox_ori = self.pull_item(idx)

        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']

        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox,
                                                                                                                   dtype=np.float32)


