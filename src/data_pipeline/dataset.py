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
  └── stego/    ← label 1 (hidden data present)

STUDENT A TASK: This is used in Weeks 3-4 once your data is ready.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


# ─── Dataset Class ────────────────────────────────────────────────────────────

class SteganalysisDataset(Dataset):
    """
    Loads clean and stego images from two folders.
    Returns (image_tensor, label) pairs.

    label = 0 → clean image (no hidden data)
    label = 1 → stego image (hidden data present)
    """

    def __init__(
        self,
        clean_dir: str,
        stego_dir: str,
        transform=None,
        max_images: Optional[int] = None
    ):
        """
        Args:
            clean_dir  : path to folder with clean images (label=0)
            stego_dir  : path to folder with stego images (label=1)
            transform  : torchvision transforms to apply to each image
            max_images : limit total images loaded (useful for quick tests)
        """
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []   # (image_path, label)

        extensions = {".png", ".jpg", ".jpeg", ".pgm", ".bmp"}

        # Collect clean images (label=0)
        clean_path = Path(clean_dir)
        clean_files = sorted([
            f for f in clean_path.iterdir()
            if f.suffix.lower() in extensions
        ])

        # Collect stego images (label=1)
        stego_path = Path(stego_dir)
        stego_files = sorted([
            f for f in stego_path.iterdir()
            if f.suffix.lower() in extensions
        ])

        # Balance the dataset (50/50 split is CRITICAL to avoid bias)
        n = min(len(clean_files), len(stego_files))
        if max_images is not None:
            n = min(n, max_images // 2)

        for f in clean_files[:n]:
            self.samples.append((str(f), 0))    # label 0 = clean

        for f in stego_files[:n]:
            self.samples.append((str(f), 1))    # label 1 = stego

        print(f"Dataset loaded: {n} clean + {n} stego = {2 * n} total images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load as grayscale (important — SRM filters work on grayscale)
        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: convert to tensor [0, 1] range
            img = T.ToTensor()(img)

        return img, label


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def get_transforms(image_size: int = 256, augment: bool = True):
    """
    Returns train and validation transforms.

    For steganalysis, we keep augmentation MINIMAL — flipping is fine,
    but rotation or color jitter would destroy subtle stego signals.
    """
    # Validation / test — no augmentation, just resize and normalize
    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])      # grayscale: single channel
    ])

    if not augment:
        return val_transform, val_transform

    # Training — light augmentation only (horizontal flip is safe)
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),          # safe: doesn't affect stego signal
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_transform


def get_dataloaders(
    clean_dir: str,
    stego_dir: str,
    batch_size: int = 32,
    image_size: int = 256,
    val_split: float = 0.2,
    max_images: Optional[int] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        clean_dir  : folder with clean images
        stego_dir  : folder with stego images
        batch_size : images per batch
        image_size : resize all images to this square size
        val_split  : fraction of data for validation (default 20%)
        max_images : limit total images (useful for quick experiments)
        num_workers: parallel data loading workers

    Returns:
        (train_loader, val_loader) tuple
    """
    train_transform, val_transform = get_transforms(image_size, augment=True)

    # Full dataset (with train transform first, we'll fix val below)
    full_dataset = SteganalysisDataset(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        transform=train_transform,
        max_images=max_images
    )

    # Split into train and validation
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )

    # Apply no-augmentation transform to validation set
    # (We do this by wrapping it)
    val_dataset.dataset.transform = val_transform   # note: this changes the full dataset
    # Better approach: create two separate datasets
    val_dataset_clean = SteganalysisDataset(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        transform=val_transform,
        max_images=max_images
    )
    _, val_dataset_proper = random_split(
        val_dataset_clean,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset_proper,
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

    train_tf, val_tf = get_transforms()
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
