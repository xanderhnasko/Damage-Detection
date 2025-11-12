import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(csv_path, output_dir=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = ["epoch", "train_loss", "test_loss", "test_accuracy"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = df["epoch"]
    
    # training loss
    axes[0].plot(epochs, df["train_loss"], 'b-', label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # test loss
    axes[1].plot(epochs, df["test_loss"], 'r-', label='Test Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Test Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # test accuracy
    axes[2].plot(epochs, df["test_accuracy"], 'g-', label='Test Accuracy', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('Test Accuracy', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylim([0, 1])  
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot training/testing curves from metrics CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/metrics/metrics.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None
    )
    args = parser.parse_args()
    
    plot_metrics(args.csv, args.output)

if __name__ == "__main__":
    main()

