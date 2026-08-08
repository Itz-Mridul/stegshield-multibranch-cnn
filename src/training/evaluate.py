"""
evaluate.py
===========
Evaluation, Metrics, and Visualisation

WHAT THIS DOES:
  Loads a trained model and produces:
  1. Confusion matrix (TP/FP/TN/FN)
  2. ROC curve with AUC score
  3. Precision, Recall, F1-score
  4. Comparison table (your model vs Xu-Net baseline)
  5. Ablation study table (what happens without each branch)
  6. Training history plots

HOW TO RUN:
  # Evaluate the full fusion model:
  python src/training/evaluate.py --checkpoint weights/fusion_best.pt --mode fusion

  # Evaluate Branch A only:
  python src/training/evaluate.py --checkpoint weights/branch_a_best.pt --mode branch_a

  # Run ablation study (compares branches individually vs fusion):
  python src/training/evaluate.py --ablation

STUDENT NOTE:
  The output of this script becomes Section 5 (Results) of your paper.
  The comparison table is your main proof of contribution.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")       # non-interactive backend — works on Colab and servers
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from training.config import Config
from data_pipeline.dataset import get_dataloaders
from branches.branch_a_pixel import BranchAClassifier
from branches.branch_b_dct import BranchBClassifier
from branches.branch_c_stats import BranchCClassifier
from model.fusion_model import MultiBranchSteganalyzer


# ─── Load Model ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, mode: str, cfg: Config):
    """
    Load a trained model from a checkpoint file.
    """
    if mode == "branch_a":
        model = BranchAClassifier(feature_dim=cfg.feature_dim_a, num_classes=cfg.num_classes)
    elif mode == "branch_b":
        model = BranchBClassifier(feature_dim=cfg.feature_dim_b, num_classes=cfg.num_classes)
    elif mode == "branch_c":
        model = BranchCClassifier(feature_dim=cfg.feature_dim_c, num_classes=cfg.num_classes)
    else:
        model = MultiBranchSteganalyzer(
            feature_dim_a=cfg.feature_dim_a,
            feature_dim_b=cfg.feature_dim_b,
            feature_dim_c=cfg.feature_dim_c
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device: torch.device):
    """
    Run model on all data in loader.

    Returns:
        all_labels     : true labels (numpy array)
        all_preds      : predicted class indices (numpy array)
        all_stego_probs: probability of being stego, for ROC curve (numpy array)
    """
    model.to(device)
    model.eval()

    all_labels, all_preds, all_stego_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)

        preds       = logits.argmax(dim=1).cpu().numpy()
        stego_probs = probs[:, 1].cpu().numpy()      # P(stego)
        labels_np   = labels.numpy()

        all_labels.extend(labels_np)
        all_preds.extend(preds)
        all_stego_probs.extend(stego_probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_stego_probs)
    )


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray, stego_probs: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.
    """
    acc       = accuracy_score(labels, preds) * 100
    precision = precision_score(labels, preds, average="binary", zero_division=0) * 100
    recall    = recall_score(labels, preds, average="binary", zero_division=0) * 100
    f1        = f1_score(labels, preds, average="binary", zero_division=0) * 100

    fpr, tpr, _ = roc_curve(labels, stego_probs)
    roc_auc     = auc(fpr, tpr) * 100

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auc":       roc_auc,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "confusion_matrix": cm.tolist()
    }


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Print metrics table."""
    print(f"\n{'=' * 50}")
    print(f"  Results: {model_name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1']:.2f}%")
    print(f"  AUC:       {metrics['auc']:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"             Predicted Clean  Predicted Stego")
    print(f"  True Clean     {metrics['tn']:5d}            {metrics['fp']:5d}")
    print(f"  True Stego     {metrics['fn']:5d}            {metrics['tp']:5d}")
    print(f"{'=' * 50}")


# ─── Plotting Functions ───────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, model_name: str, save_path: str):
    """
    Save confusion matrix as a heatmap image.
    """
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["CLEAN", "STEGO"],
        yticklabels=["CLEAN", "STEGO"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {save_path}")


def plot_roc_curve(results_dict: dict, save_path: str):
    """
    Plot ROC curves for multiple models on the same chart.

    Args:
        results_dict: {model_name: metrics_dict}
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]
    for i, (name, metrics) in enumerate(results_dict.items()):
        fpr = np.array(metrics["fpr"])
        tpr = np.array(metrics["tpr"])
        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{name} (AUC = {metrics['auc']:.1f}%)"
        )

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 50%)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ ROC curve saved → {save_path}")


def plot_training_history(history_path: str, save_path: str):
    """
    Plot train/val loss and accuracy curves from saved JSON history.
    """
    if not os.path.exists(history_path):
        print(f"No history file found at {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#E74C3C")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#3498DB")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#E74C3C")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   color="#3498DB")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Training history plot saved → {save_path}")


def print_comparison_table(results: dict):
    """
    Print the comparison table that goes in your paper.
    """
    print("\n" + "=" * 72)
    print("  COMPARISON TABLE (for your paper — Section 6)")
    print("=" * 72)
    print(f"  {'Model':<28} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 72)
    for name, m in results.items():
        print(f"  {name:<28} {m['accuracy']:>9.2f}% {m['precision']:>9.2f}% "
              f"{m['recall']:>7.2f}% {m['f1']:>7.2f}% {m['auc']:>7.2f}%")
    print("=" * 72)
    print("  Note: All results on BOSS benchmark. Fill in Xu-Net baseline.")


def run_ablation_study(cfg: Config, loader):
    """
    Run ablation study: evaluate each branch individually.
    This shows that EACH branch contributes to the final result.
    Required for paper acceptance.
    """
    print("\n=== Ablation Study ===")
    ablation_results = {}
    device = cfg.device

    checkpoints = {
        "Branch A (Pixel CNN)":   (cfg.branch_a_name, "branch_a"),
        "Branch B (DCT CNN)":     (cfg.branch_b_name, "branch_b"),
        "Branch C (Stats MLP)":   (cfg.branch_c_name, "branch_c"),
        "Full Fusion (A+B+C)":    (cfg.best_model_name, "fusion"),
    }

    for model_name, (ckpt_name, mode) in checkpoints.items():
        ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  [{model_name}] checkpoint not found: {ckpt_path} — skipping")
            continue

        model = load_model(ckpt_path, mode, cfg)
        labels, preds, stego_probs = run_inference(model, loader, device)
        metrics = compute_metrics(labels, preds, stego_probs)
        ablation_results[model_name] = metrics
        print(f"  [{model_name}] Acc: {metrics['accuracy']:.2f}%  AUC: {metrics['auc']:.2f}%")

    if ablation_results:
        print_comparison_table(ablation_results)

    return ablation_results


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained steganalysis model")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--mode", default="fusion",
                        choices=["branch_a", "branch_b", "branch_c", "fusion"],
                        help="Which model type to evaluate")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study (all branches vs fusion)")
    parser.add_argument("--history", type=str,
                        help="Path to training history JSON to plot")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()

    # Data loaders
    _, val_loader = get_dataloaders(
        clean_dir=cfg.clean_dir,
        stego_dir=cfg.stego_dir,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        max_images=args.max_images,
    )

    os.makedirs(cfg.results_dir, exist_ok=True)
    all_results = {}

    if args.ablation:
        # Run full ablation study
        ablation_results = run_ablation_study(cfg, val_loader)
        all_results.update(ablation_results)
    elif args.checkpoint:
        # Evaluate a single model
        model = load_model(args.checkpoint, args.mode, cfg)
        labels, preds, stego_probs = run_inference(model, val_loader, cfg.device)
        metrics = compute_metrics(labels, preds, stego_probs)
        model_name = f"{args.mode.upper()} Model"
        print_metrics(metrics, model_name)

        plot_confusion_matrix(
            metrics, model_name,
            save_path=os.path.join(cfg.results_dir, f"confusion_matrix_{args.mode}.png")
        )
        all_results[model_name] = metrics

    # Plot ROC curves for all evaluated models
    if all_results:
        plot_roc_curve(
            all_results,
            save_path=os.path.join(cfg.results_dir, "roc_curves.png")
        )

    # Plot training history
    if args.history:
        plot_training_history(
            history_path=args.history,
            save_path=os.path.join(cfg.results_dir, "training_history.png")
        )


if __name__ == "__main__":
    main()
