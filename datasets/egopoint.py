# datasets/egopoint.py
import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

class EgoPointDataset(Dataset):
    """
    EgoPoint jsonl -> OneRef eval pipeline adapter
    Each line:
      {
        "image_id": ...,
        "image_path": ...,
        "expression": ...,
        "gt_bbox_xywh": [x,y,w,h],
        ...
      }
    """
    def __init__(self, jsonl_path, transform=None, max_query_len=64):
        self.jsonl_path = Path(jsonl_path)
        self.transform = transform
        self.max_query_len = max_query_len

        self.samples = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_path = s["image_path"]
        text = s["expression"]

        # ---- load image ----
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # ---- gt box ----
        gt_xyxy = xywh_to_xyxy(s["gt_bbox_xywh"])
        gt_xyxy = torch.tensor(gt_xyxy, dtype=torch.float32).unsqueeze(0)  # [1,4]

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,                     # tensor
            "text": text,                     
            "boxes": gt_xyxy,                 # [N,4] in xyxy, original scale
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "image_id": s.get("image_id", str(idx)),
        }
