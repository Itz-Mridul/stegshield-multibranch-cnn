"""
branch_a_pixel.py
=================
Branch A: Pixel Domain CNN

WHAT THIS DOES:
  Inspects the raw pixel values of an image to detect LSB (Least Significant
  Bit) steganography.

  Architecture pipeline:
    Input Image (grayscale) → SRM Filters → Pixel Residual Maps → CNN → Feature Vector

HOW IT WORKS:
  1. SRM filters amplify tiny pixel changes caused by steganography
  2. CNN learns to classify these amplified residuals
  3. Outputs a 256-dimensional feature vector (not a final class yet!)
     The fusion model will combine this with Branches B and C.

TARGET ACCURACY (standalone): 85%+ on LSB steganography

STUDENT A TASK: Train this branch in Weeks 5-6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.srm_filters import SRMFilterLayer


class PixelCNN(nn.Module):
    """
    Branch A: Pixel Domain Convolutional Neural Network.

    Takes a grayscale image, applies SRM filters to get residuals,
    then runs a CNN to extract a feature vector for classification.

    Architecture:
        SRMFilterLayer (30 channels, fixed)
          ↓
        ConvBlock 1: Conv(30→64) → BN → ReLU → MaxPool
          ↓
        ConvBlock 2: Conv(64→128) → BN → ReLU → MaxPool
          ↓
        ConvBlock 3: Conv(128→256) → BN → ReLU → AvgPool
          ↓
        GlobalAveragePool → Flatten → FC(256) → feature_vector
    """

    def __init__(self, feature_dim: int = 256):
        """
        Args:
            feature_dim: size of the output feature vector (default: 256)
                         This is what the fusion layer will receive.
        """
        super().__init__()

        self.feature_dim = feature_dim

        # ── Fixed SRM filters (not trained) ──────────────────────────────────
        self.srm = SRMFilterLayer()    # outputs 30 residual channels

        # ── CNN Block 1: extract low-level stego patterns ─────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(30, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)      # spatial: H/2, W/2
        )

        # ── CNN Block 2: extract mid-level patterns ───────────────────────────
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)      # spatial: H/4, W/4
        )

        # ── CNN Block 3: extract high-level stego signatures ──────────────────
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)      # spatial: H/8, W/8
        )

        # ── Global Average Pooling: collapse spatial dims ─────────────────────
        # Output shape: (batch, 256) regardless of input image size
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Fully Connected → feature vector ─────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, 1, H, W) — grayscale images

        Returns:
            feature vector of shape (batch, feature_dim)
        """
        # Step 1: Apply SRM filters to get pixel residuals
        x = self.srm(x)             # (batch, 30, H, W)

        # Step 2: Run through CNN blocks
        x = self.conv1(x)           # (batch, 64, H/2, W/2)
        x = self.conv2(x)           # (batch, 128, H/4, W/4)
        x = self.conv3(x)           # (batch, 256, H/8, W/8)

        # Step 3: Global average pooling + flatten
        x = self.gap(x)             # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)

        # Step 4: FC layer → feature vector
        x = self.fc(x)             # (batch, feature_dim)

        return x


class BranchAClassifier(nn.Module):
    """
    Standalone Branch A model with its own classification head.
    Use this for INDEPENDENT training and evaluation of Branch A.
    Later, the PixelCNN part will be plugged into the fusion model.
    """

    def __init__(self, feature_dim: int = 256, num_classes: int = 2):
        """
        Args:
            feature_dim: feature vector size from the CNN
            num_classes: 2 (clean vs stego)
        """
        super().__init__()
        self.backbone = PixelCNN(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor (batch, 1, H, W)

        Returns:
            logits of shape (batch, 2)
            Use F.softmax() to get probabilities, or CrossEntropyLoss directly.
        """
        features = self.backbone(x)          # (batch, feature_dim)
        logits = self.classifier(features)   # (batch, 2)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features only (for use in fusion model).
        """
        return self.backbone(x)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Branch A (Pixel CNN)...")

    # Create a dummy batch of 4 grayscale images, 256×256
    batch = torch.randn(4, 1, 256, 256)

    # Standalone classifier test
    model = BranchAClassifier(feature_dim=256, num_classes=2)
    model.eval()

    with torch.no_grad():
        logits = model(batch)
        features = model.get_features(batch)

    print(f"Input shape:    {batch.shape}")
    print(f"Feature shape:  {features.shape}  ← this goes into fusion model")
    print(f"Output shape:   {logits.shape}    ← (batch, 2) for clean/stego")
    print(f"Sample logits:  {logits[0]}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}  (SRM filters are fixed)")
    print("\n✅ Branch A OK!")
