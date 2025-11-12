"""
Train ResNet18 classifier on xView2 damage detection crops.

Reads a manifest CSV with image paths and bounding boxes, trains a ResNet18 model
to classify building damage into 4 categories (no-damage, minor, major, destroyed).
When using a config file, trains on train_events and tests/validates on test_events.
Saves best and latest checkpoints based on test accuracy.
"""
import argparse, os, random, csv
import torch, torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from datasets import ThreeChannelDataset
import yaml

def load_config(config_path):
    """Load YAML config file"""
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
        test_events = events['test_events']  # Use test_events for validation during training
        print(f"[CONFIG] Training on events: {train_events}")
        print(f"[CONFIG] Testing/validating on events: {test_events}")

    # Create train/test datasets from manifest CSV
    # When config is provided, uses event-based filtering (no 90/10 split)
    train_ds = ThreeChannelDataset(
        args.manifest, 
        split="train", 
        img_size=args.img_size,
        allowed_events=train_events
    )
    test_ds = ThreeChannelDataset(
        args.manifest, 
        split="test",  
        img_size=args.img_size,
        allowed_events=test_events
    )


    ### STEP 2: MAKE BATCHES USING COLLATE FUNCTION ###
    # images of dif sizes (different croppings) can't be stacked into single tensor, need to collate

    def collate_train(batch):
        imgs, ys = zip(*batch)  # Unpack (image, label) pairs
        # for training: resize, random flip, normalize (see src/datasets.py)
        imgs = torch.stack([train_ds.tf_train(im) for im in imgs])
        ys   = torch.tensor(ys).long() 
        return imgs, ys
    def collate_eval(batch):
        imgs, ys = zip(*batch)
        imgs = torch.stack([test_ds.tf_eval(im) for im in imgs])
        ys   = torch.tensor(ys).long()
        return imgs, ys

    # Create data loaders with above collations
    # Reduced num_workers and added pin_memory for better GPU performance
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=2, pin_memory=True, collate_fn=collate_train)
    test_dl  = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_eval)

    ### STEP 3: INIT MODEL ###
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")
    if device.type == 'cuda':
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # init ResNet18 with ImageNet pretrained weights
    model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    # Original ImageNet1k has 1000 classes, we only need 4
    # replace fully connected final layer to map 512 features -> 4 classes
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)  
    
    # using AdamW optimizer instead of SGD 
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cost = nn.CrossEntropyLoss().to(device)

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)
    best = 0.0  # Track best test accuracy
    
    # Initialize CSV metrics file
    metrics_file = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_accuracy"])


    ### STEP 4: TRAIN MODEL ###

    for epoch in range(args.epochs):
        #TRAINING PHASE
        model.train()  
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True) # move to device

            opt.zero_grad()  # Clear gradients from previous iter
            loss = cost(model(xb), yb)  # Forward pass + compute loss
            loss.backward()  # Backward pass 
            opt.step()  # Update weights
            tr_loss += loss.item() 
        
        #TESTING PHASE
        model.eval() 
        correct = total = 0
        test_loss = 0.0
        with torch.no_grad():  # not updating gradients here
            for xb, yb in test_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True) # move to device
                output = model(xb)
                test_loss += cost(output, yb).item()
                pred = output.argmax(1)  # get predicted class
                is_correct = (pred == yb)
                correct += is_correct.sum().item()  # num correct predictions
                total += len(yb)  
        
        #test accuracy 
        acc = correct / total if total else 0.0
        
        # average losses for each epoch
        avg_train_loss = tr_loss / len(train_dl)
        avg_test_loss = test_loss / len(test_dl)
        
        # Log metrics to CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_test_loss, acc])
        
        # Save latest checkpoint 
        torch.save(model.state_dict(), os.path.join(args.out_dir, "latest.pt"))
        
        # Save best checkpoint if test accuracy improved
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
        
        print(f"epoch: {epoch+1}/{args.epochs} loss={avg_train_loss:.4f} test_loss={avg_test_loss:.4f} test_acc={acc:.3f}")

    print(f"[DONE] best_test_acc={best:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config file for event filtering (e.g., configs/hurricanes.yaml)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    main(args)
