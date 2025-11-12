"""PyTorch classes for loading X-channel RGB images with bounding box crops (right now, only 3-channel model)
    Reads a CSV manifest and sets up train/eval splits and applies image transforms"""

import csv
from PIL import Image
import torch
import torchvision as tv
from pathlib import Path
from collections import OrderedDict

# Get mean and std from ResNet18 ImageNet-1K weights metadata
# used to match the preprocessing used during ImageNet training
weights = tv.models.ResNet18_Weights.IMAGENET1K_V1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class ThreeChannelDataset(torch.utils.data.Dataset):

    def __init__(self, manifest_csv, split="train", split_frac=0.9, img_size=224, allowed_events=None):
        rows = []
        # Read all rows from the manifest CSV 
        with open(manifest_csv) as f:
           rows = list(csv.DictReader(f))

        # Filter by allowed events if specified
        if allowed_events is not None:
            def extract_event_name(img_path):
                filename = Path(img_path).name
                # Event name is everything before the first "_"
                event_name = filename.split('_')[0]
                return event_name
            
            rows = [r for r in rows if extract_event_name(r["img_path"]) in allowed_events]
            if not rows:
                raise ValueError(f"No rows found matching events: {allowed_events}")
            # When using event-based filtering, use all rows (no further splitting)
            self.rows = rows
        else:
            # Only apply 90-10 split when not using event-based filtering
            split_idx = int(len(rows)*split_frac)
            self.rows = rows[:split_idx] if split=="train" else rows[split_idx:]
        
        self.split = split
        self.img_size = img_size
        # Cache for storing recently loaded images 
        self.cache = OrderedDict()
        self.cache_max = 128

        # training transforms
        self.tf_train = tv.transforms.Compose([
            tv.transforms.Resize((img_size, img_size)),  # Resize to target size
            tv.transforms.RandomHorizontalFlip(), # random augmentation
            tv.transforms.ToTensor(),  # convert to PyTorch tensor 
            tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # normalize to ImageNet pretrained stats
        ])
        # eval transform (no augmentation)
        self.tf_eval = tv.transforms.Compose([
            tv.transforms.Resize((img_size, img_size)),  
            tv.transforms.ToTensor(),  
            tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), 
        ])

    def __len__(self):
        return len(self.rows)

    def _get_cached_image(self, path: str):
        if path in self.cache:
            im = self.cache.pop(path)   
            self.cache[path] = im
            return im
        im = Image.open(path).convert("RGB")
        if len(self.cache) >= self.cache_max:
            self.cache.popitem(last=False) 
        self.cache[path] = im
        return im

    def __getitem__(self, i):
        r = self.rows[i]
        im = self._get_cached_image(r["img_path"])
        xmin,ymin,xmax,ymax = map(int, [r["xmin"],r["ymin"],r["xmax"],r["ymax"]])
        crop = im.crop((xmin,ymin,xmax,ymax))
        
        # Apply transforms based on split
        if self.split == "train":
            crop = self.tf_train(crop)
        else:
            crop = self.tf_eval(crop)
        
        y = int(r["label_id"])
        return crop, y

