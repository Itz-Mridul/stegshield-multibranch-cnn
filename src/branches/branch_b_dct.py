"""
branch_b_dct.py
===============
Branch B: Frequency Domain CNN

WHAT THIS DOES:
  Inspects the FREQUENCY content of an image using DCT (Discrete Cosine Transform)
  to detect JPEG-domain steganography.

  Architecture pipeline:
    Input Image (grayscale) → Block DCT (8×8) → DCT Feature Map → CNN → Feature Vector

WHY DCT?
  JPEG compression works using DCT. Steganographers often hide data by
  modifying DCT coefficients slightly — the same math JPEG uses to compress.
  By converting the image to DCT space first, we can see these modifications
  directly. Human eyes looking at the image see nothing. Our CNN sees everything.

HOW DCT WORKS (simplified):
  - Divide image into 8×8 pixel blocks (same as JPEG does)
  - Apply 2D DCT on each block → converts pixels to frequency components
  - Low frequencies = broad patterns; High frequencies = fine details
  - Steganography adds tiny distortions in high frequencies
  - CNN learns to detect these distortions

TARGET ACCURACY (standalone): 80%+ on JPEG-domain steganography

STUDENT B TASK: Train this branch in Weeks 5-6 (parallel to Student A).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dctn


# ─── DCT Feature Extraction ───────────────────────────────────────────────────

def compute_block_dct(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Apply 2D DCT on non-overlapping blocks of the image.
    This is exactly how JPEG compression works.

    Args:
        image     : grayscale numpy array of shape (H, W), float32
        block_size: block size for DCT (JPEG standard = 8)

    Returns:
        DCT coefficient map of shape (H, W) — same spatial size as input.
        Each 8×8 region contains DCT coefficients instead of pixel values.
    """
    H, W = image.shape
    dct_map = np.zeros_like(image, dtype=np.float32)

    # Process image in 8×8 blocks
    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            block = image[i:i+block_size, j:j+block_size].astype(np.float32)

            # Apply 2D DCT using scipy
            dct_block = dctn(block, type=2, norm='ortho')

            dct_map[i:i+block_size, j:j+block_size] = dct_block

    return dct_map


def image_to_dct_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of image tensors to DCT feature maps.
    Called once during data loading (or can be a transform).

    Args:
        image_tensor: shape (batch, 1, H, W), values in [-1, 1]

    Returns:
        DCT tensor of shape (batch, 1, H, W)
    """
    batch_size = image_tensor.shape[0]
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    dct_batch = []

    for b in range(batch_size):
        # Convert to numpy, range [0, 255]
        img_np = image_tensor[b, 0].cpu().numpy()
        img_np = (img_np + 1.0) * 127.5   # [-1,1] → [0,255]

        # Compute block DCT
        dct_map = compute_block_dct(img_np)

        # Normalize DCT coefficients (they can be very large)
        # Log compression to keep values in a reasonable range
        dct_map = np.sign(dct_map) * np.log1p(np.abs(dct_map))

        dct_batch.append(dct_map)

    # Stack back to tensor: (batch, H, W) → (batch, 1, H, W)
    dct_array = np.stack(dct_batch, axis=0)[:, np.newaxis, :, :]
    return torch.tensor(dct_array, dtype=torch.float32)


# ─── DCT CNN Model ────────────────────────────────────────────────────────────

class FrequencyCNN(nn.Module):
    """
    Branch B: Frequency Domain Convolutional Neural Network.

    Takes a DCT coefficient map (same spatial size as original image)
    and learns to detect frequency-domain steganography anomalies.

    Architecture:
        DCT Map (1 channel)
          ↓
        ConvBlock 1: Conv(1→32) → BN → ReLU → MaxPool
          ↓
        ConvBlock 2: Conv(32→64) → BN → ReLU → MaxPool
          ↓
        ConvBlock 3: Conv(64→128) → BN → ReLU → MaxPool
          ↓
        ConvBlock 4: Conv(128→256) → BN → ReLU → AvgPool
          ↓
        GlobalAveragePool → Flatten → FC(256) → feature_vector
    """

    def __init__(self, feature_dim: int = 256):
        """
        Args:
            feature_dim: size of output feature vector (default: 256)
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Block 1: learn basic frequency patterns
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # H/2, W/2
        )

        # Block 2: learn composite frequency patterns
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # H/4, W/4
        )

        # Block 3: learn stego-specific frequency anomalies
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # H/8, W/8
        )

        # Block 4: high-level frequency representation
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Global average pooling — handles any input size
        self.gap = nn.AdaptiveAvgPool2d(1)

        # FC layer → feature vector
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: DCT feature map tensor of shape (batch, 1, H, W)

        Returns:
            feature vector of shape (batch, feature_dim)
        """
        x = self.conv1(x)           # (batch, 32, H/2, W/2)
        x = self.conv2(x)           # (batch, 64, H/4, W/4)
        x = self.conv3(x)           # (batch, 128, H/8, W/8)
        x = self.conv4(x)           # (batch, 256, H/8, W/8)
        x = self.gap(x)             # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        x = self.fc(x)             # (batch, feature_dim)
        return x


class BranchBClassifier(nn.Module):
    """
    Standalone Branch B model with its own classification head.
    Input: RAW IMAGE tensor (this class applies DCT internally).
    Use this for independent training and evaluation.
    """

    def __init__(self, feature_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.backbone = FrequencyCNN(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw image tensor (batch, 1, H, W)

        Returns:
            logits (batch, 2)
        """
        # Convert to DCT feature map
        dct_map = image_to_dct_tensor(x).to(x.device)

        features = self.backbone(dct_map)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion model.
        """
        dct_map = image_to_dct_tensor(x).to(x.device)
        return self.backbone(dct_map)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Branch B (DCT Frequency CNN)...")

    batch = torch.randn(4, 1, 256, 256)

    model = BranchBClassifier(feature_dim=256, num_classes=2)
    model.eval()

    with torch.no_grad():
        logits = model(batch)
        features = model.get_features(batch)

    print(f"Input shape:    {batch.shape}")
    print(f"Feature shape:  {features.shape}  ← this goes into fusion model")
    print(f"Output shape:   {logits.shape}    ← (batch, 2) for clean/stego")
    print(f"Sample logits:  {logits[0]}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\n✅ Branch B OK!")
