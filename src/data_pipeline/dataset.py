"""
dataset.py
==========
PyTorch Dataset for Steganography Detection

WHAT THIS DOES:
  Wraps the clean/ and stego/ image folders into a proper PyTorch Dataset
  that the training loop can use with a DataLoader.

STRUCTURE EXPECTED:
  data/
  ├── clean/    ← label 0 (no hidden data)
  └── stego/    ← label 1 (hidden data present), SAME filenames as clean/

CRITICAL DESIGN DECISIONS (v2 — bug-fixed):

  Decision 1 — Pair-safe train/val split:
    The split is done on SOURCE IMAGE NAMES, not on individual samples.
    If "img_001.png" is in the validation set, BOTH data/clean/img_001.png
    AND data/stego/img_001.png end up in validation.
    This is mandatory because a clean/stego pair carries shared scene content.
    If they land in different splits, the model can learn to distinguish
    training scenes from validation scenes — inflating accuracy by 15-20%.

  Decision 2 — Lossless preprocessing (NO bilinear resize for LSB):
    Bilinear interpolation averages neighbouring pixel values.
    This destroys the 1-bit LSB payload:
      pixel 200 (clean) and 201 (stego) both become 200.5 → rounded to 200.
      The label becomes meaningless.
    SOLUTION: Random crops for training, centre crops for validation.
    This preserves exact pixel values. Images must be at least crop_size×crop_size.
    For 512×512 BOSS images with crop_size=256 — works perfectly.
    For smaller images: use the full image (no crop) and reject via assert.

STUDENT A TASK: This is used in Weeks 3-4 once your data is ready.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Set

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ─── Dataset Class ────────────────────────────────────────────────────────────

class SteganalysisDataset(Dataset):
    """
    Loads clean and stego images from two folders.
    Returns (image_tensor, label) pairs.

    label = 0 → clean image (no hidden data)
    label = 1 → stego image (hidden data present)

    IMPORTANT: stego/ filenames must exactly match clean/ filenames.
    The pair-safe split in get_dataloaders() depends on this.
    """

    def __init__(
        self,
        clean_dir: str,
        stego_dir: str,
        allowed_stems: Optional[Set[str]] = None,
        transform=None,
    ):
        """
        Args:
            clean_dir    : path to folder with clean images (label=0)
            stego_dir    : path to folder with stego images (label=1)
            allowed_stems: set of file stems (without extension) to include.
                           If None, all files are included.
                           Pass train_stems or val_stems from the split helper.
            transform    : torchvision transforms to apply to each image
        """
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []   # (image_path, label)

        extensions = {".png", ".pgm", ".bmp"}
        # Note: .jpg / .jpeg intentionally excluded — JPEG re-saves will
        # corrupt LSB payload (see lsb_embedder.py). Clean inputs may be
        # JPEG; stego outputs are always saved as PNG.

        clean_path = Path(clean_dir)
        stego_path = Path(stego_dir)

        # Collect only files whose stem is in allowed_stems (if provided)
        clean_files = sorted([
            f for f in clean_path.iterdir()
            if f.suffix.lower() in extensions
            and (allowed_stems is None or f.stem in allowed_stems)
        ])
        stego_files = sorted([
            f for f in stego_path.iterdir()
            if f.suffix.lower() in extensions
            and (allowed_stems is None or f.stem in allowed_stems)
        ])

        # Build paired sample list — warn if sizes differ
        n_clean = len(clean_files)
        n_stego = len(stego_files)
        if n_clean != n_stego:
            print(f"[WARNING] clean ({n_clean}) and stego ({n_stego}) counts differ. "
                  f"Using min={min(n_clean, n_stego)}. Check dataset setup.")
        n = min(n_clean, n_stego)

        for f in clean_files[:n]:
            self.samples.append((str(f), 0))

        for f in stego_files[:n]:
            self.samples.append((str(f), 1))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid images found in:\n"
                f"  clean: {clean_dir}\n"
                f"  stego: {stego_dir}\n"
                "Accepted formats: PNG, PGM, BMP (JPEG excluded — LSB is lossless only).\n"
                "Run: bash scripts/setup_boss_dataset.sh  to populate the dataset."
            )
        else:
            print(f"Dataset: {n} clean + {n} stego = {2 * n} total images"
                  + (f" [from {len(allowed_stems)} source stems]" if allowed_stems else ""))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load as grayscale — SRM filters operate on grayscale only
        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, label


# ─── Lossless Transforms ──────────────────────────────────────────────────────

def get_transforms(crop_size: int = 256):
    """
    Returns (train_transform, val_transform).

    NO RESIZE — uses random/centre crops instead to preserve pixel values.

    For LSB detection, even a single pixel value rounding from interpolation
    can flip a true positive to a false negative. Crops guarantee that every
    pixel reaching the model is exactly the value set by embed_lsb().

    Augmentation policy:
      - RandomCrop: shifts which pixels are seen, does NOT alter pixel values.
      - RandomHorizontalFlip: LSB is independent of spatial position → safe.
      - NO colour jitter, rotation, or scaling — these all alter pixel values.

    Args:
        crop_size: size of the square crop (default 256).
                   Images must be at least crop_size × crop_size.
                   BOSS dataset (512×512) works perfectly.
    """
    # Validation: deterministic centre crop
    val_transform = T.Compose([
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])      # scale to [-1, 1]
    ])

    # Training: random crop + horizontal flip
    train_transform = T.Compose([
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_transform


# ─── Pair-Safe Split Helper ───────────────────────────────────────────────────

def split_stems_by_pair(
    clean_dir: str,
    val_split: float = 0.20,
    max_images: Optional[int] = None,
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Split image NAMES (stems) into train and val sets.

    The split is done on stems BEFORE creating Dataset objects.
    Both the clean and stego Dataset will use the same stem sets,
    guaranteeing that a clean/stego pair is NEVER split across train/val.

    Args:
        clean_dir : folder with clean images (used only to enumerate stems)
        val_split : fraction of stems to hold out for validation
        max_images: if set, limits total pairs considered
        seed      : random seed for reproducibility

    Returns:
        (train_stems, val_stems) — two sets of file stem strings
    """
    extensions = {".png", ".pgm", ".bmp"}
    clean_path = Path(clean_dir)

    all_stems = sorted([
        f.stem for f in clean_path.iterdir()
        if f.suffix.lower() in extensions
    ])

    if max_images is not None:
        all_stems = all_stems[:max_images]

    rng = np.random.default_rng(seed)
    rng.shuffle(all_stems)

    n_val = max(1, int(len(all_stems) * val_split))
    val_stems = set(all_stems[:n_val])
    train_stems = set(all_stems[n_val:])

    print(f"Pair-safe split: {len(train_stems)} train stems, {len(val_stems)} val stems "
          f"({len(val_stems) / (len(train_stems) + len(val_stems)) * 100:.1f}% val)")

    return train_stems, val_stems


# ─── DataLoader Builder ───────────────────────────────────────────────────────

def get_dataloaders(
    clean_dir: str,
    stego_dir: str,
    batch_size: int = 32,
    crop_size: int = 256,
    val_split: float = 0.2,
    max_images: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders with a PAIR-SAFE split.

    Args:
        clean_dir  : folder with clean images
        stego_dir  : folder with stego images (same filenames as clean)
        batch_size : images per batch
        crop_size  : crop size (NOT resize — preserves LSB pixel values)
        val_split  : fraction held out for validation
        max_images : limit total pairs considered (for quick experiments)
        num_workers: parallel data loading workers
        seed       : random seed for reproducibility

    Returns:
        (train_loader, val_loader) tuple
    """
    train_transform, val_transform = get_transforms(crop_size)

    # Step 1: Split on stems (pair-safe)
    train_stems, val_stems = split_stems_by_pair(
        clean_dir=clean_dir,
        val_split=val_split,
        max_images=max_images,
        seed=seed
    )

    # Step 2: Create SEPARATE Dataset objects for train and val
    #         Each gets its own transform — NO shared transform object mutation.
    train_dataset = SteganalysisDataset(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        allowed_stems=train_stems,
        transform=train_transform,
    )
    val_dataset = SteganalysisDataset(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        allowed_stems=val_stems,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True       # drop last incomplete batch for stable BN stats
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="data/clean", help="Clean images directory")
    parser.add_argument("--stego", default="data/stego", help="Stego images directory")
    parser.add_argument("--verify", action="store_true", help="Print sample info")
    parser.add_argument("--samples", type=int, default=5, help="Samples to print")
    args = parser.parse_args()

    _, val_tf = get_transforms()
    dataset = SteganalysisDataset(args.clean, args.stego, transform=val_tf)

    print(f"\nTotal images in dataset: {len(dataset)}")
    print(f"Class balance: {sum(1 for _, l in dataset.samples if l == 0)} clean, "
          f"{sum(1 for _, l in dataset.samples if l == 1)} stego")

    if args.verify:
        print(f"\nFirst {args.samples} samples:")
        for i in range(min(args.samples, len(dataset))):
            img, label = dataset[i]
            label_name = "CLEAN" if label == 0 else "STEGO"
            print(f"  [{i}] {label_name} | shape: {img.shape} | "
                  f"min: {img.min():.3f} max: {img.max():.3f}")
