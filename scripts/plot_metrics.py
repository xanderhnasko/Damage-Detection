#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_training_metrics(csv_path):
    df = pd.read_csv(csv_path)

    # Backward compat: handle old column names
    val_loss_col = 'val_loss' if 'val_loss' in df.columns else 'test_loss'
    val_acc_col = 'val_accuracy' if 'val_accuracy' in df.columns else 'test_accuracy'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = df['epoch'].astype(int)
    
    ax1.plot(epochs, df['train_loss'], linewidth=2, label='Training Loss')
    ax1.plot(epochs, df[val_loss_col], linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss', fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    ax2.plot(epochs, df[val_acc_col], linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Val Accuracy', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    min_acc = df[val_acc_col].min()
    max_acc = df[val_acc_col].max()
    margin = (max_acc - min_acc) * 0.1
    ax2.set_ylim(min_acc - margin, max_acc + margin)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_metrics(sys.argv[1])
