"""
train.py
========
Main Training Loop

WHAT THIS DOES:
  Trains any of the branch models (A, B, C) or the full fusion model.
  Supports Google Colab GPU training out of the box.

HOW TO RUN:

  # Train Branch A alone (Student A, Week 5-6):
  python src/training/train.py --mode branch_a

  # Train Branch B alone (Student B, Week 5-6):
  python src/training/train.py --mode branch_b

  # Train Branch C alone (both students, Week 7):
  python src/training/train.py --mode branch_c

  # Train full fusion model (Week 9-10):
  python src/training/train.py --mode fusion

  # Quick test with small dataset (to verify setup):
  python src/training/train.py --mode branch_a --max_images 200 --epochs 2

COLAB USAGE:
  Upload this project to Google Drive, then in Colab:
    !python src/training/train.py --mode fusion --epochs 50

STUDENT NOTE:
  - Training on Colab with 10,000 images takes ~2-4 hours per branch
  - Save your checkpoints — they go in the weights/ folder
  - Download the weights/ folder to your laptop after training
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from training.config import Config
from data_pipeline.dataset import get_dataloaders
from branches.branch_a_pixel import BranchAClassifier
from branches.branch_b_dct import BranchBClassifier
from branches.branch_c_stats import BranchCClassifier
from model.fusion_model import MultiBranchSteganalyzer


# ─── Training Loop ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set random seeds for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(cfg: Config):
    """
    Create the correct model based on training mode.
    """
    if cfg.mode == "branch_a":
        return BranchAClassifier(feature_dim=cfg.feature_dim_a, num_classes=cfg.num_classes)
    elif cfg.mode == "branch_b":
        return BranchBClassifier(feature_dim=cfg.feature_dim_b, num_classes=cfg.num_classes)
    elif cfg.mode == "branch_c":
        return BranchCClassifier(feature_dim=cfg.feature_dim_c, num_classes=cfg.num_classes)
    elif cfg.mode in ("fusion", "joint"):
        model = MultiBranchSteganalyzer(
            feature_dim_a=cfg.feature_dim_a,
            feature_dim_b=cfg.feature_dim_b,
            feature_dim_c=cfg.feature_dim_c,
            num_classes=cfg.num_classes
        )
        # Try to load pre-trained branch weights (for "fusion" mode)
        if cfg.mode == "fusion":
            model.load_pretrained_branches(
                branch_a_path=os.path.join(cfg.checkpoint_dir, cfg.branch_a_name),
                branch_b_path=os.path.join(cfg.checkpoint_dir, cfg.branch_b_name),
                branch_c_path=os.path.join(cfg.checkpoint_dir, cfg.branch_c_name)
            )
        return model
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. "
                         f"Choose from: branch_a, branch_b, branch_c, fusion, joint")


def get_checkpoint_name(cfg: Config) -> str:
    name_map = {
        "branch_a": cfg.branch_a_name,
        "branch_b": cfg.branch_b_name,
        "branch_c": cfg.branch_c_name,
        "fusion":   cfg.best_model_name,
        "joint":    cfg.best_model_name,
    }
    return os.path.join(cfg.checkpoint_dir, name_map[cfg.mode])


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler = None
) -> dict:
    """
    Run one full training epoch.

    Returns:
        dict with 'loss' and 'accuracy' for this epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training (if GPU available)
        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Track metrics
        total_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += images.size(0)

        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            batch_acc = (predicted == labels).float().mean().item() * 100
            print(f"  Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}  Acc: {batch_acc:.1f}%")

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples * 100
    return {"loss": avg_loss, "accuracy": avg_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Run validation and return loss + accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples * 100
    return {"loss": avg_loss, "accuracy": avg_acc}


def train(cfg: Config):
    """
    Main training function.
    """
    set_seed(cfg.seed)
    cfg.summary()
    device = cfg.device
    print(f"\nTraining on: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        clean_dir=cfg.clean_dir,
        stego_dir=cfg.stego_dir,
        batch_size=cfg.batch_size,
        crop_size=cfg.image_size,      # lossless crop, not bilinear resize
        val_split=cfg.val_split,
        max_images=cfg.max_images,
        num_workers=cfg.num_workers
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError(
            f"Empty dataloader(s): train={len(train_loader)} batches, "
            f"val={len(val_loader)} batches. "
            "Check that data/clean/ and data/stego/ contain PNG/PGM images, "
            "then re-run: bash scripts/setup_boss_dataset.sh"
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(cfg).to(device)
    print(f"\nModel: {cfg.mode.upper()} — "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    if cfg.optimizer == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )

    # ── Learning Rate Scheduler ───────────────────────────────────────────────
    # Guard: if epochs <= warmup_epochs (e.g. quick test runs), CosineAnnealingLR
    # gets T_max=0 which causes ZeroDivisionError. Clamp to safe values.
    effective_warmup = min(cfg.warmup_epochs, max(0, cfg.epochs - 1))
    cosine_epochs    = max(1, cfg.epochs - effective_warmup)

    if effective_warmup > 0:
        warmup    = LinearLR(optimizer, start_factor=0.1, total_iters=effective_warmup)
        cosine    = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[effective_warmup])
    else:
        # No warmup — just cosine from epoch 1
        scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)

    # ── Mixed Precision (GPU only) ────────────────────────────────────────────
    scaler = GradScaler() if (cfg.mixed_precision and device.type == "cuda") else None

    # ── Training Loop ─────────────────────────────────────────────────────────
    # Early stopping monitors VALIDATION LOSS (lower = better).
    # We do NOT use val_accuracy because accuracy can plateau while loss still
    # improves (important for calibration), and accuracy is less smooth near
    # saturation. Saving the checkpoint at lowest val_loss gives the best
    # generalising model.
    best_val_loss = float("inf")
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    checkpoint_path = get_checkpoint_name(cfg)

    print(f"\nStarting training for {cfg.epochs} epochs...")
    print(f"Early stopping: patience={cfg.early_stop_patience} epochs (monitors val_loss)")
    print("-" * 60)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        # Validate
        val_metrics   = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0

        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"Epoch [{epoch:3d}/{cfg.epochs}] "
              f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1f}% | "
              f"Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.1f}% | "
              f"LR={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        # Save best checkpoint (by val_loss, not val_acc)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ★ New best model saved → {checkpoint_path} "
                  f"(val_loss={best_val_loss:.4f}, val_acc={val_metrics['accuracy']:.1f}%)")
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping on val_loss
        if no_improve >= cfg.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(val_loss did not improve for {no_improve} epochs)")
            break

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {checkpoint_path}")

    # Save training history
    history_path = os.path.join(cfg.results_dir, f"history_{cfg.mode}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    return history, best_val_loss


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train steganography detection model")
    parser.add_argument("--mode", default="fusion",
                        choices=["branch_a", "branch_b", "branch_c", "fusion", "joint"],
                        help="Which model to train")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--max_images",  type=int,   default=None,
                        help="Limit total images (set 200 for quick test)")
    parser.add_argument("--clean_dir",   type=str,   default=None)
    parser.add_argument("--stego_dir",   type=str,   default=None)
    parser.add_argument("--no_gpu",      action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.mode = args.mode
    if args.epochs:     cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.learning_rate = args.lr
    if args.max_images: cfg.max_images = args.max_images
    if args.clean_dir:  cfg.clean_dir = args.clean_dir
    if args.stego_dir:  cfg.stego_dir = args.stego_dir
    if args.no_gpu:     cfg.use_gpu = False

    train(cfg)


if __name__ == "__main__":
    main()
