#!/usr/bin/env python3
"""
Train ResNet18/50 classifier on crops.
Reads  manifest CSV with image paths and bboxes, trains a ResNet model
to classify building damage into 4 categories (no-damage, minor, major, destroyed),
optionally with a background class when unmatched detections are kept.
"""
import argparse, os, random, csv, time
import torch, torchvision as tv
import torchvision.transforms.v2 as tvv2
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torch.amp import autocast, GradScaler
from torchvision.transforms import InterpolationMode
from datasets import ThreeChannelDataset, IMAGENET_MEAN, IMAGENET_STD

def gpu_mem_fmt(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return f"alloc={alloc:.2f}G reserved={reserved:.2f}G peak={peak:.2f}G"

class GroupedByImageSampler(Sampler):
    """ indices grouped by source image"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.groups = {}
        for idx, row in enumerate(getattr(dataset, "rows", [])):
            self.groups.setdefault(row["img_path"], []).append(idx)

    def __iter__(self):
        # shuffle group order each epoch
        group_keys = list(self.groups.keys())
        random.shuffle(group_keys)
        for k in group_keys:
            grp = self.groups[k]
            random.shuffle(grp)
            for idx in grp:
                yield idx

    def __len__(self):
        return len(self.dataset)


def main(args):

    ### STEP 1: GATHER TRAIN, VAL, TEST DATA SETS ###

    # fixed seeds for now
    torch.manual_seed(0); random.seed(0)

    if not args.train_manifest or not args.val_manifest:
        raise ValueError("train_manifest and val_manifest must be provided (detector-generated manifests).")

    background_label = None if args.background_label is not None and args.background_label < 0 else args.background_label
    train_ds = ThreeChannelDataset(
        args.train_manifest,
        split=None,
        img_size=args.img_size,
        cache_size=256,
        background_label=background_label,
    )
    val_ds = ThreeChannelDataset(
        args.val_manifest,
        split=None,
        img_size=args.img_size,
        cache_size=256,
        background_label=background_label,
    )
    test_ds = None
    if args.test_manifest:
        test_ds = ThreeChannelDataset(
            args.test_manifest,
            split=None,
            img_size=args.img_size,
            cache_size=256,
            background_label=background_label,
        )
    print(f"Using manifests: train={args.train_manifest}, val={args.val_manifest}, test={args.test_manifest or 'None'}")
    
    # Determine number of classes (handles optional background)
    n_list = [train_ds.n_classes, val_ds.n_classes]
    if test_ds:
        n_list.append(test_ds.n_classes)
    num_classes = max(n_list)
    if num_classes == 0:
        raise ValueError("No samples found in datasets after filtering labels.")
    print(f"Detected {num_classes} classes (including background={background_label})")

    #  class weights for imbalanced classes
    class_counts = [0] * num_classes
    for row in getattr(train_ds, "rows", []):
        class_counts[int(row["label_id"])] += 1
    total_samples = sum(class_counts)
    class_weights = [
        (total_samples / (num_classes * count)) if count > 0 else 0.0
        for count in class_counts
    ]
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class counts: {class_counts}")


    ### STEP 2: DATA LOADERS ###
    # Transforms applied in dataset.py
    torch.backends.cudnn.benchmark = True 
    workers = min(6, os.cpu_count() or 6)
    prefetch = 6
    train_sampler = GroupedByImageSampler(train_ds)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )
    val_dl  = DataLoader(val_ds,  batch_size=args.bs, shuffle=False, drop_last=False, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch)
    test_dl = None
    if test_ds:
        test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch)

    print("train_dl.num_workers =", getattr(train_dl, "num_workers", "N/A"))
    print("val_dl.num_workers  =", getattr(val_dl, "num_workers", "N/A"))
    if test_dl:
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
    # Original ImageNet1k has 1000 classes, we only need num_classes (optionally incl. background)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)  
    # Disable torch.compile for now to reduce potential peak memory; re-enable if stable

    # using SGD with momentum for fine-tuning 
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    cost = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05).to(device)
    
    # Mixed precision training for speed
    scaler = GradScaler()

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)
   
    
    # Initialize CSV metrics file
    metrics_file = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])


    ### STEP 4: TRAIN MODEL ###

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    best_path = os.path.join(args.out_dir, "best.pt")
    latest_path = os.path.join(args.out_dir, "latest.pt")
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

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
        
        #VALIDATION PHASE
        model.eval() 
        val_correct = val_total = 0
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():  # not updating gradients here
            for xb, yb in val_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True) # move to device
                xb = tf_eval_gpu(xb)
                
                with autocast(device_type=device.type):
                    output = model(xb)
                    loss = cost(output, yb)
                
                bs = yb.size(0)
                val_loss += loss.item() * bs
                val_seen += bs
                
                pred = output.argmax(1)  # get predicted class
                is_correct = (pred == yb)
                val_correct += is_correct.sum().item()  # num correct predictions
                val_total += len(yb)  
        
        #val accuracy 
        val_acc = val_correct / val_total if val_total else 0.0
        
        # avg losses for each epoch
        avg_train_loss = tr_loss / tr_seen
        avg_val_loss = val_loss / val_seen
        
        # write metrics to CSV for graphing
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_acc])
        
        # Save checkpoint 
        torch.save(model.state_dict(), latest_path)
        
        prev_lr = opt.param_groups[0]["lr"]

        # Save best checkpoint if val loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1

        # LR schedule on plateau (val loss)
        scheduler.step(avg_val_loss)
        new_lr = opt.param_groups[0]["lr"]
        lr_changed = new_lr != prev_lr
        
        num_batches = max(1, len(train_dl))
        epoch_time = time.perf_counter() - epoch_start
        mem_info = gpu_mem_fmt(device)
        print(
            f"epoch: {epoch+1}/{args.epochs} "
            f"train loss = {avg_train_loss:.4f} "
            f"val loss = {avg_val_loss:.4f} "
            f"val acc. = {val_acc:.3f} "
            f"[lr: {new_lr:.6f}"
            f"; epoch_time: {epoch_time:.1f}s, data_time/batch: {data_time/num_batches:.3f}s, batch_time: {batch_time/num_batches:.3f}s"
            f"{', ' + mem_info if mem_info else ''}]"
        )
        if lr_changed:
            print(f"  lr reduced from {prev_lr:.6f} to {new_lr:.6f} (val plateau)")

        if args.patience and no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1}; best val loss {best_val_loss:.4f}, best val acc {best_val_acc:.3f} (epoch {best_epoch})")
            break

    # Final test evaluation on the best checkpoint
    if test_dl:
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state)
        model.eval()
        test_correct = test_total = 0
        test_loss, test_seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                xb = tf_eval_gpu(xb)
                with autocast(device_type=device.type):
                    output = model(xb)
                    loss = cost(output, yb)
                bs = yb.size(0)
                test_loss += loss.item() * bs
                test_seen += bs
                pred = output.argmax(1)
                test_correct += (pred == yb).sum().item()
                test_total += len(yb)
        avg_test_loss = test_loss / test_seen
        test_acc = test_correct / test_total if test_total else 0.0
        print(f"[TEST] loss={avg_test_loss:.4f} acc={test_acc:.3f} (best val loss={best_val_loss:.4f}, best val acc={best_val_acc:.3f}, epoch {best_epoch})")
    else:
        print(f"best val loss = {best_val_loss:.4f}, best val acc = {best_val_acc:.3f} (epoch {best_epoch})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_manifest", type=str, required=True, help="Train manifest (detector outputs)")
    ap.add_argument("--val_manifest", type=str, required=True, help="Val manifest (detector outputs)")
    ap.add_argument("--test_manifest", type=str, default=None, help="Optional test manifest for final eval only (detector outputs)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=10, help="Early stop after N epochs without val loss improvement (0 disables)")
    ap.add_argument("--background_label", type=int, default=4, help="Map unmatched detector boxes (<0) to this class id; set to -1 to drop them")
    args = ap.parse_args()
    main(args)
