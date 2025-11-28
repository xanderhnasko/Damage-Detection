#!/usr/bin/env python3
"""
Train ResNet18/50 classifier on crops.
Reads  manifest CSV with image paths and bboxes, trains a ResNet model
to classify building damage into 4 categories (no-damage, minor, major, destroyed).
"""
import argparse, os, random, csv, time
import torch, torchvision as tv
import torchvision.transforms.v2 as tvv2
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision.transforms import InterpolationMode
from datasets import ThreeChannelDataset, IMAGENET_MEAN, IMAGENET_STD
import yaml

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return {
        'train_events': config.get('train_events', []),
        'val_events': config.get('val_events', []),
        'test_events': config.get('test_events', [])
    }

def main(args):

    ### STEP 1: GATHER TRAIN AND TEST DATA SETS ###

    # fixed seeds for now
    torch.manual_seed(0); random.seed(0)
    # event filtering if available
    train_events = None
    test_events = None
    if args.config:
        events = load_config(args.config)
        train_events = events['train_events']
        test_events = events['test_events'] 
        print(f"Training on events: {train_events}")
        print(f"Testing/validating on events: {test_events}")

    # Dataset options for caching/preload
    ds_kwargs = dict(
        img_size=args.img_size,
        cache_size=256,     
        cache_all=False,      # hold all decoded images 
        preload=False,       
        sort_by_img=True,    # group by source image to maximize cache hits
    )

    # Create train/test datasets from manifest CSV
    # When config is provided, uses event-based filtering (no 90/10 split)
    train_ds = ThreeChannelDataset(
        args.manifest, 
        split="train", 
        allowed_events=train_events,
        **ds_kwargs,
    )
    test_ds = ThreeChannelDataset(
        args.manifest, 
        split="test",  
        allowed_events=test_events,
        **ds_kwargs,
    )
    
    #  class weights for imbalanced classes
    class_counts = [0] * 4
    for row in getattr(train_ds, "rows", []):
        class_counts[int(row["label_id"])] += 1
    total_samples = sum(class_counts)
    class_weights = [
        (total_samples / (4 * count)) if count > 0 else 0.0
        for count in class_counts
    ]
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class counts: {class_counts}")


    ### STEP 2: DATA LOADERS ###
    # Transforms applied in dataset.py
    torch.backends.cudnn.benchmark = True 
    workers = min(10, os.cpu_count() or 10)
    prefetch = 8
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  drop_last=True, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch)
    test_dl  = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, drop_last=False, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch)

    print("train_dl.num_workers =", getattr(train_dl, "num_workers", "N/A"))
    print("test_dl.num_workers  =", getattr(test_dl, "num_workers", "N/A"))

    # GPU-side augmentation pipelines
    # Light GPU aug: resize already done on CPU for collation; avoid double resize
    tf_train_gpu = tvv2.Compose([
        tvv2.RandomHorizontalFlip(p=0.5),
        tvv2.RandomRotation(5, interpolation=InterpolationMode.BILINEAR, fill=0),
        tvv2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_eval_gpu = tvv2.Compose([
        tvv2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    ### STEP 3: INIT MODEL ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # init ResNet18/50 with ImageNet pretrained weights
    model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    # Original ImageNet1k has 1000 classes, we only need 4
    # replace fully connected final layer to map 512 features -> 4 classes
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)  
    try:
        model = torch.compile(model)  # PyTorch 2.x: optimize compute; disable if unsupported
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    # using SGD with momentum for fine-tuning 
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    cost = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05).to(device)
    
    # Mixed precision training for speed
    scaler = GradScaler()

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)
   
    
    # Initialize CSV metrics file
    metrics_file = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_accuracy"])


    ### STEP 4: TRAIN MODEL ###

    best = 0.0  # Track best test accuracy
    for epoch in range(args.epochs):
        #TRAINING PHASE
        model.train()  
        tr_loss, tr_seen = 0.0, 0
        data_time = 0.0
        batch_time = 0.0
        end = time.perf_counter()

        for xb, yb in train_dl:
            data_time += time.perf_counter() - end
            batch_start = time.perf_counter()
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True) # move to GPU
            xb = tf_train_gpu(xb)

            opt.zero_grad(set_to_none=True)  # Clear gradients from previous iter
            
            # Mixed precision forward pass
            with autocast(device_type=device.type):
                logits = model(xb)
                loss = cost(logits, yb)  # Forward pass + compute loss
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(opt) # gradient step
            scaler.update() #update gradient
            
            bs = yb.size(0)
            tr_loss += loss.item() * bs 
            tr_seen += bs 
            batch_time += time.perf_counter() - batch_start
            end = time.perf_counter()
        
        #TESTING PHASE
        model.eval() 
        correct = total = 0
        test_loss, test_seen = 0.0, 0
        with torch.no_grad():  # not updating gradients here
            for xb, yb in test_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True) # move to device
                xb = tf_eval_gpu(xb)
                
                with autocast(device_type=device.type):
                    output = model(xb)
                    loss = cost(output, yb)
                
                bs = yb.size(0)
                test_loss += loss.item() * bs
                test_seen += bs
                
                pred = output.argmax(1)  # get predicted class
                is_correct = (pred == yb)
                correct += is_correct.sum().item()  # num correct predictions
                total += len(yb)  
        
        #test accuracy 
        acc = correct / total if total else 0.0
        
        # avg losses for each epoch
        avg_train_loss = tr_loss / tr_seen
        avg_test_loss = test_loss / test_seen
        
        # write metrics to CSV for graphing
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_test_loss, acc])
        
        # Save checkpoint 
        torch.save(model.state_dict(), os.path.join(args.out_dir, "latest.pt"))
        
        # Save best checkpoint if test accuracy improved
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
        
        num_batches = max(1, len(train_dl))
        print(
            f"epoch: {epoch+1}/{args.epochs} "
            f"train loss = {avg_train_loss:.4f} "
            f"test loss = {avg_test_loss:.4f} "
            f"test acc. = {acc:.3f} "
            f"[data_time/batch: {data_time/num_batches:.3f}s, batch_time: {batch_time/num_batches:.3f}s]"
        )

    print(f"best test acc. = {best:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", type=str, default=None) # Path to config file for event filtering (e.g., configs/hurricanes.yaml)"
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=320)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    main(args)
