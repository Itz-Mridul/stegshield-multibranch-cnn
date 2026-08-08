"""
quantize.py
===========
INT8 Quantization Pipeline — CPU-Only Deployment

WHAT THIS DOES:
  Takes a trained model (which uses 32-bit floating point weights)
  and compresses it to INT8 (8-bit integers).

  Results:
    - Model size: ~4× smaller (e.g., 80MB → 20MB)
    - Inference speed: ~2-3× faster on CPU
    - Accuracy loss: expected 1-5% (report the real number honestly)

WHY THIS MATTERS:
  DRDO endpoints are air-gapped — no internet, no GPU server.
  They run standard Intel/AMD CPU machines.
  INT8 quantization makes our model small enough and fast enough
  to run on these CPU-only machines without needing any GPU.

HONEST NOTE (from project docs):
  "Budget for 2-5% accuracy drop." Don't assume < 2%.
  Measure it with your data and report the real number.

HOW TO RUN:
  python src/deployment/quantize.py \\
    --checkpoint weights/fusion_best.pt \\
    --output weights/fusion_int8.pt

STUDENT A TASK: Run this in Week 11.
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.quantization

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from training.config import Config
from model.fusion_model import MultiBranchSteganalyzer


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (total_params + total_buffers) / 1024 / 1024


def measure_inference_time(model: nn.Module, device: torch.device, n_runs: int = 50) -> float:
    """
    Measure average inference time on a single image (in milliseconds).
    Uses CPU always (even if GPU is available) because we're testing deployment.
    """
    model.eval()
    dummy = torch.randn(1, 1, 256, 256)   # single image, CPU

    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)

    # Time it
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy)
    end = time.perf_counter()

    avg_ms = (end - start) / n_runs * 1000
    return avg_ms


def quantize_model(
    model: nn.Module,
    output_path: str,
    compare_accuracy: bool = False,
    val_loader=None,
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    """
    Apply INT8 dynamic quantization to the model.

    Dynamic quantization:
      - Does NOT require a calibration dataset
      - Works on Linear layers (fully connected layers)
      - Is the standard first approach for deployment
      - Quantizes weights to INT8; activations quantized at runtime

    Args:
        model           : trained FP32 model
        output_path     : where to save the INT8 model
        compare_accuracy: if True, run accuracy check on val_loader
        val_loader      : DataLoader for accuracy comparison
        device          : CPU (quantization must run on CPU)

    Returns:
        INT8 quantized model
    """
    # ── Move to CPU ───────────────────────────────────────────────────────────
    model = model.cpu()
    model.eval()

    print("\n" + "=" * 55)
    print("  INT8 Quantization Pipeline")
    print("=" * 55)

    # ── Measure before ────────────────────────────────────────────────────────
    size_before = get_model_size_mb(model)
    time_before = measure_inference_time(model, torch.device("cpu"))

    print(f"\n  BEFORE quantization:")
    print(f"    Model size:       {size_before:.2f} MB")
    print(f"    Inference time:   {time_before:.1f} ms/image (CPU)")

    # ── Apply Dynamic INT8 Quantization ───────────────────────────────────────
    # We quantize the linear (dense) layers — these dominate model size
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},     # quantize all Linear layers
        dtype=torch.qint8             # INT8 format
    )

    # ── Measure after ─────────────────────────────────────────────────────────
    size_after  = get_model_size_mb(quantized_model)
    time_after  = measure_inference_time(quantized_model, torch.device("cpu"))
    compression = size_before / size_after if size_after > 0 else 0

    print(f"\n  AFTER INT8 quantization:")
    print(f"    Model size:       {size_after:.2f} MB")
    print(f"    Inference time:   {time_after:.1f} ms/image (CPU)")
    print(f"\n  Compression ratio: {compression:.2f}×")
    print(f"  Speed improvement: {time_before / time_after:.2f}×")

    # ── Accuracy Comparison (optional) ────────────────────────────────────────
    if compare_accuracy and val_loader is not None:
        print("\n  Comparing accuracy (FP32 vs INT8)...")
        fp32_acc = _quick_accuracy(model, val_loader)
        int8_acc = _quick_accuracy(quantized_model, val_loader)
        drop = fp32_acc - int8_acc

        print(f"    FP32 accuracy: {fp32_acc:.2f}%")
        print(f"    INT8 accuracy: {int8_acc:.2f}%")
        print(f"    Accuracy drop: {drop:.2f}%  "
              f"({'✅ within budget' if drop < 5 else '⚠️ above 5% — check pipeline'})")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(quantized_model.state_dict(), output_path)
    print(f"\n  ✅ INT8 model saved → {output_path}")
    print("=" * 55)

    return quantized_model


@torch.no_grad()
def _quick_accuracy(model, loader) -> float:
    """Quick accuracy check on a DataLoader."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total * 100


def load_quantized_model(path: str, cfg: Config) -> nn.Module:
    """
    Load and return an INT8 quantized model for CPU inference.
    """
    model = MultiBranchSteganalyzer(
        feature_dim_a=cfg.feature_dim_a,
        feature_dim_b=cfg.feature_dim_b,
        feature_dim_c=cfg.feature_dim_c
    )

    # Re-apply quantization config (needed to load INT8 state dict)
    model = torch.quantization.quantize_dynamic(
        model.cpu(),
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize trained model to INT8")
    parser.add_argument("--checkpoint", required=True,
                        help="FP32 model checkpoint path")
    parser.add_argument("--output",     required=True,
                        help="Output path for INT8 model")
    parser.add_argument("--compare",    action="store_true",
                        help="Compare FP32 vs INT8 accuracy on validation set")
    args = parser.parse_args()

    cfg = Config()

    # Load trained FP32 model
    model = MultiBranchSteganalyzer(
        feature_dim_a=cfg.feature_dim_a,
        feature_dim_b=cfg.feature_dim_b,
        feature_dim_c=cfg.feature_dim_c
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    val_loader = None
    if args.compare:
        from data_pipeline.dataset import get_dataloaders
        _, val_loader = get_dataloaders(
            clean_dir=cfg.clean_dir,
            stego_dir=cfg.stego_dir,
            batch_size=16,
            image_size=cfg.image_size,
            max_images=500    # only 500 images for quick accuracy check
        )

    quantize_model(
        model=model,
        output_path=args.output,
        compare_accuracy=args.compare,
        val_loader=val_loader
    )


if __name__ == "__main__":
    main()
