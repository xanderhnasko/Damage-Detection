 """
 PyTorch classes for loading X-channel RGB images with bounding box crops (right now, only 3-channel model)
    
 Reads a CSV manifest and sets up train/eval splits and applies image transforms
 """

import csv
from PIL import Image
import torch
import torchvision as tv

# Get mean and std from ResNet18 ImageNet-1K weights metadata
# used to match the preprocessing used during ImageNet training
weights = tv.models.ResNet18_Weights.IMAGENET1K_V1
IMAGENET_MEAN = weights.meta["mean"]
IMAGENET_STD = weights.meta["std"]

class ThreeChannelDataset(torch.utils.data.Dataset):

    def __init__(self, manifest_csv, split="train", split_frac=0.9, img_size=224):

        rows = []
        # Read all rows from the manifest CSV 
        with open(manifest_csv) as f:
           rows = list(csv.DictReader(f))

        # create train and eval datasets (90-10 default)
        split_idx = int(len(rows)*split_frac)
        self.rows = rows[:split_idx] if split=="train" else rows[split_idx:]
        self.img_size = img_size
        # Cache for storing recently loaded images 
        self.cache = {}

        # training transforms
        self.tf_train = tv.transforms.Compose([
            tv.transforms.Resize((img_size, img_size)),  # Resize to target size
            tv.transforms.RandomHorizontalFlip(), # random augmentation
            tv.transforms.ToTensor(),  # convert to PyTorch tensor (0-1 range)
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

    def _load_image(self, p):
        # Return cached image if available
        if p in self.cache: return self.cache[p]
        # Load image and convert to RGB 
        im = Image.open(p).convert("RGB")
        if len(self.cache) > 64: self.cache.clear()
        self.cache[p] = im
        return im

    def __getitem__(self, i):
        r = self.rows[i]
        im = self._load_image(r["img_path"])
        xmin,ymin,xmax,ymax = map(int, [r["xmin"],r["ymin"],r["xmax"],r["ymax"]])
        crop = im.crop((xmin,ymin,xmax,ymax))
        y = int(r["label_id"])
        return crop, y

