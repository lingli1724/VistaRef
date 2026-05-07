import math
import torch
import random
from PIL import Image, ImageEnhance, ImageFilter

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_utils import xyxy2xywh
from utils.misc import interpolate, mdetr_interpolate


def crop(image, box, region, obj_mask=None, kp_root=None, kp_tip=None):
    cropped_image = F.crop(image, *region)

    i, j, h, w = region

    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_box = box - torch.as_tensor([j, i, j, i], dtype=torch.float32)
    cropped_box = torch.min(cropped_box.reshape(2, 2), max_size)
    cropped_box = cropped_box.clamp(min=0)
    cropped_box = cropped_box.reshape(-1)

    if obj_mask is not None:
        obj_mask = obj_mask[:, i:i + h, j:j + w]

    if kp_root is not None:
        kp_root = kp_root - torch.as_tensor([j, i], dtype=torch.float32)
        kp_root = torch.clamp(kp_root, min=0)
        kp_root[0] = torch.clamp(kp_root[0], max=float(w))
        kp_root[1] = torch.clamp(kp_root[1], max=float(h))

    if kp_tip is not None:
        kp_tip = kp_tip - torch.as_tensor([j, i], dtype=torch.float32)
        kp_tip = torch.clamp(kp_tip, min=0)
        kp_tip[0] = torch.clamp(kp_tip[0], max=float(w))
        kp_tip[1] = torch.clamp(kp_tip[1], max=float(h))

    return cropped_image, cropped_box, obj_mask, kp_root, kp_tip


def resize_according_to_long_side(img, box, size, kp_root=None, kp_tip=None):
    h, w = img.height, img.width
    ratio = float(size / float(max(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    img = F.resize(img, (new_h, new_w))
    box = box * ratio

    if kp_root is not None:
        kp_root = kp_root * ratio
    if kp_tip is not None:
        kp_tip = kp_tip * ratio

    return img, box, kp_root, kp_tip


def resize_according_to_short_side(img, box, size, kp_root=None, kp_tip=None):
    h, w = img.height, img.width
    ratio = float(size / float(min(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    img = F.resize(img, (new_h, new_w))
    box = box * ratio

    if kp_root is not None:
        kp_root = kp_root * ratio
    if kp_tip is not None:
        kp_tip = kp_tip * ratio

    return img, box, kp_root, kp_tip


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict):
        for t in self.transforms:
            input_dict = t(input_dict)
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomBrightness(object):
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1-self.brightness, 1+self.brightness)
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img
        

class RandomContrast(object):
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1-self.contrast, 1+self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img
        

class RandomSaturation(object):
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation
    
    def __call__(self, img):
        saturation_factor = random.uniform(1-self.saturation, 1+self.saturation)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast   = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, input_dict):
        if random.random() < 0.8:
            image = input_dict['img']
            func_inds = list(np.random.permutation(3))
            for func_id in func_inds:
                if func_id == 0:
                    image = self.rand_brightness(image)
                elif func_id == 1:
                    image = self.rand_contrast(image)
                elif func_id == 2:
                    image = self.rand_saturation(image)
            input_dict['img'] = image

        return input_dict


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.], aug_blur=False):
        self.sigma = sigma
        self.p = 0.5 if aug_blur else 0.
    
    def __call__(self, input_dict):
        if random.random() < self.p:
            img = input_dict['img']
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            input_dict['img'] = img

        return input_dict


class RandomHorizontalFlip(object):
    def __call__(self, input_dict):
        if random.random() < 0.5:
            img = input_dict['img']
            box = input_dict['box']
            text = input_dict['text']

            img = F.hflip(img)
            text = text.replace('right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')
            h, w = img.height, img.width
            box = box[[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1], dtype=torch.float32) + \
                  torch.as_tensor([w, 0, w, 0], dtype=torch.float32)

            input_dict['img'] = img
            input_dict['box'] = box
            input_dict['text'] = text

            if 'kp_root' in input_dict:
                kp_root = input_dict['kp_root']
                kp_root = kp_root.clone()
                kp_root[0] = w - kp_root[0]
                input_dict['kp_root'] = kp_root

            if 'kp_tip' in input_dict:
                kp_tip = input_dict['kp_tip']
                kp_tip = kp_tip.clone()
                kp_tip[0] = w - kp_tip[0]
                input_dict['kp_tip'] = kp_tip

        return input_dict


class RandomResize(object):
    def __init__(self, sizes, with_long_side=True):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.with_long_side = with_long_side

    def __call__(self, input_dict):
        img = input_dict['img']
        box = input_dict['box']
        size = random.choice(self.sizes)

        kp_root = input_dict['kp_root'] if 'kp_root' in input_dict else None
        kp_tip = input_dict['kp_tip'] if 'kp_tip' in input_dict else None

        if self.with_long_side:
            resized_img, resized_box, kp_root, kp_tip = resize_according_to_long_side(img, box, size, kp_root, kp_tip)
        else:
            resized_img, resized_box, kp_root, kp_tip = resize_according_to_short_side(img, box, size, kp_root, kp_tip)

        input_dict['img'] = resized_img
        input_dict['box'] = resized_box

        if kp_root is not None:
            input_dict['kp_root'] = kp_root
        if kp_tip is not None:
            input_dict['kp_tip'] = kp_tip

        new_img_size = (resized_img.height, resized_img.width)

        if "obj_mask" in input_dict:
            input_dict["obj_mask"] = mdetr_interpolate(
                input_dict["obj_mask"][:, None].float(), new_img_size, mode="nearest"
            )[:, 0] > 0.5

        return input_dict
        

class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, max_try: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try = max_try

    def __call__(self, input_dict):
        img = input_dict['img']
        box = input_dict['box']
        obj_mask = input_dict['obj_mask'] if "obj_mask" in input_dict else None
        kp_root = input_dict['kp_root'] if "kp_root" in input_dict else None
        kp_tip = input_dict['kp_tip'] if "kp_tip" in input_dict else None

        num_try = 0
        while num_try < self.max_try:
            num_try += 1
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])  # [i, j, h, w]

            box_xywh = xyxy2xywh(box)
            box_x, box_y = box_xywh[0], box_xywh[1]
            if box_x > region[0] and box_y > region[1]:
                img, box, obj_mask, kp_root, kp_tip = crop(
                    img, box, region, obj_mask=obj_mask, kp_root=kp_root, kp_tip=kp_tip
                )
                input_dict['img'] = img
                input_dict['box'] = box
                if obj_mask is not None:
                    input_dict['obj_mask'] = obj_mask
                if kp_root is not None:
                    input_dict['kp_root'] = kp_root
                if kp_tip is not None:
                    input_dict['kp_tip'] = kp_tip
                return input_dict

        return input_dict


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
    
    def __call__(self, input_dict):
        text = input_dict['text']
        
        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        for wd in dir_words:
            if wd in text:
                return self.transforms1(input_dict)

        if random.random() < self.p:
            return self.transforms2(input_dict)
        else:
            return self.transforms1(input_dict)


class ToTensor(object):
    def __call__(self, input_dict):
        img = input_dict['img']
        img = F.to_tensor(img)
        input_dict['img'] = img
        
        return input_dict


class NormalizeAndPad(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640, aug_translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate

    def __call__(self, input_dict):
        img = input_dict['img']
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[1:]
        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        out_img = torch.zeros((3, self.size, self.size)).float()
        out_mask = torch.ones((self.size, self.size)).int()

        out_img[:, top:top + h, left:left + w] = img
        out_mask[top:top + h, left:left + w] = 0

        input_dict['img'] = out_img
        input_dict['mask'] = out_mask

        if 'obj_mask' in input_dict.keys():
            obj_mask = torch.zeros((1, self.size, self.size)).float()
            input_dict_obj_mask = input_dict['obj_mask']
            if input_dict_obj_mask.dim() == 2:
                input_dict_obj_mask = input_dict_obj_mask[None].float()
            else:
                input_dict_obj_mask = input_dict_obj_mask.float()
            obj_mask[:, top:top + h, left:left + w] = input_dict_obj_mask
            input_dict['obj_mask'] = obj_mask

        if 'box' in input_dict.keys():
            box = input_dict['box']  # x1y1x2y2
            box = box.clone()
            box[0], box[2] = box[0] + left, box[2] + left
            box[1], box[3] = box[1] + top, box[3] + top
            H, W = out_img.shape[-2:]
            box = xyxy2xywh(box)
            box = box / torch.tensor([W, H, W, H], dtype=torch.float32)
            input_dict['box'] = box

        # -------- 新增：kp 同步 pad + normalize --------
        if 'kp_root' in input_dict:
            kp_root = input_dict['kp_root'].clone()
            kp_root[0] = kp_root[0] + left
            kp_root[1] = kp_root[1] + top
            kp_root = kp_root / torch.tensor([self.size, self.size], dtype=torch.float32)
            input_dict['kp_root'] = kp_root

        if 'kp_tip' in input_dict:
            kp_tip = input_dict['kp_tip'].clone()
            kp_tip[0] = kp_tip[0] + left
            kp_tip[1] = kp_tip[1] + top
            kp_tip = kp_tip / torch.tensor([self.size, self.size], dtype=torch.float32)
            input_dict['kp_tip'] = kp_tip

        return input_dict


class NormalizeAndPad_FOR_MIM(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640, aug_translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate

    def translate_or_pad(self, input_dict, top, left, h, w):
        img = input_dict['img']

        out_img = torch.zeros((3, self.size, self.size)).float()
        out_mask = torch.ones((self.size, self.size)).int()

        out_img[:, top:top + h, left:left + w] = img
        out_mask[top:top + h, left:left + w] = 0

        input_dict['img'] = out_img
        input_dict['mask'] = out_mask

        if 'obj_mask' in input_dict.keys():
            obj_mask = torch.zeros((1, self.size, self.size)).float()
            input_obj_mask = input_dict['obj_mask']
            if input_obj_mask.dim() == 2:
                input_obj_mask = input_obj_mask[None].float()
            else:
                input_obj_mask = input_obj_mask.float()
            obj_mask[:, top:top + h, left:left + w] = input_obj_mask
            input_dict['obj_mask'] = obj_mask

        if 'box' in input_dict.keys():
            box = input_dict['box']
            box = box.clone()
            box[0], box[2] = box[0] + left, box[2] + left
            box[1], box[3] = box[1] + top, box[3] + top
            H, W = out_img.shape[-2:]
            box = xyxy2xywh(box)
            box = box / torch.tensor([W, H, W, H], dtype=torch.float32)
            input_dict['box'] = box

        if 'kp_root' in input_dict:
            kp_root = input_dict['kp_root'].clone()
            kp_root[0] += left
            kp_root[1] += top
            kp_root = kp_root / torch.tensor([self.size, self.size], dtype=torch.float32)
            input_dict['kp_root'] = kp_root

        if 'kp_tip' in input_dict:
            kp_tip = input_dict['kp_tip'].clone()
            kp_tip[0] += left
            kp_tip[1] += top
            kp_tip = kp_tip / torch.tensor([self.size, self.size], dtype=torch.float32)
            input_dict['kp_tip'] = kp_tip

        return input_dict

    def __call__(self, input_dict):
        for_patches = input_dict.copy()
        for_visual_tokens = input_dict.copy()
        # img = input_dict['img']
        for_patches['img'] = F.normalize(for_patches['img'], mean=self.mean, std=self.std)
        h, w = for_patches['img'].shape[1:]

        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        for_patches = self.translate_or_pad(for_patches, top, left, h, w)
        for_visual_tokens = self.translate_or_pad(for_visual_tokens, top, left, h, w)

        return for_patches, for_visual_tokens


class WithoutNormAndPad(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640, aug_translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate

    def __call__(self, input_dict):
        img = input_dict['img']

        h, w = img.shape[1:]
        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        out_img = torch.zeros((3, self.size, self.size)).float()
        out_mask = torch.ones((self.size, self.size)).int()

        out_img[:, top:top + h, left:left + w] = img
        out_mask[top:top + h, left:left + w] = 0

        input_dict['img'] = out_img
        input_dict['mask'] = out_mask

        if 'obj_mask' in input_dict.keys():
            obj_mask = torch.zeros((1, self.size, self.size)).float()
            input_obj_mask = input_dict['obj_mask']
            if input_obj_mask.dim() == 2:
                input_obj_mask = input_obj_mask[None].float()
            else:
                input_obj_mask = input_obj_mask.float()
            obj_mask[:, top:top + h, left:left + w] = input_obj_mask
            input_dict['obj_mask'] = obj_mask

        if 'box' in input_dict.keys():
            box = input_dict['box']
            box = box.clone()
            box[0], box[2] = box[0] + left, box[2] + left
            box[1], box[3] = box[1] + top, box[3] + top
            H, W = out_img.shape[-2:]
            box = xyxy2xywh(box)
            box = box / torch.tensor([W, H, W, H], dtype=torch.float32)
            input_dict['box'] = box

        if 'kp_root' in input_dict:
            kp_root = input_dict['kp_root'].clone()
            kp_root[0] += left
            kp_root[1] += top
            kp_root = kp_root / torch.tensor([self.size, self.size], dtype=torch.float32)
            input_dict['kp_root'] = kp_root

        if 'kp_tip' in input_dict:
            kp_tip = input_dict['kp_tip'].clone()
            kp_tip[0] += left
            kp_tip[1] += top
            kp_tip = kp_tip / torch.tensor([self.size, self.size], dtype=torch.float32)
            input_dict['kp_tip'] = kp_tip

        return input_dict

