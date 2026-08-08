"""
branch_c_stats.py
=================
Branch C: Statistical Features MLP

WHAT THIS DOES:
  Extracts classical statistical features from an image and uses an MLP
  (Multi-Layer Perceptron) to detect statistical anomalies caused by
  steganography.

  Feature pipeline:
    Input Image → Handcrafted Statistical Features → MLP → Feature Vector

FEATURES EXTRACTED:
  1. Pixel histogram: How are pixel values distributed? (256 bins)
  2. Chi-square statistic: Are pairs of adjacent histogram bins balanced?
     (LSB embedding makes odd bins slightly heavier — chi-square detects this)
  3. Entropy: How random is the pixel distribution?
  4. Mean, standard deviation, skewness, kurtosis: Basic statistics
  5. Local variance: How does pixel variance vary across the image?

WHY THIS CATCHES STEGANOGRAPHY:
  When you embed data via LSB, you change pixel values by ±1.
  This makes histogram bins at even numbers VERY slightly different
  from bins at odd numbers.  The chi-square test is specifically designed
  to catch this pair-of-bins imbalance.

NOTE ON HONESTY (from project docs):
  Branch C works well against classical/older steganography.
  It is NOT reliable against modern adaptive steganography.
  In your report: say Branch C targets "classical statistical anomalies."
  Don't claim it catches GAN-based stego.

TARGET: Be honest. Branch C adds value as a third perspective; Branches
        A and B carry the main classification load.

BOTH STUDENTS TASK: Build this together in Week 7.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2_contingency, kurtosis as scipy_kurtosis, skew as scipy_skew
from typing import Tuple


# ─── Feature Extraction Functions ────────────────────────────────────────────

def extract_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute pixel value histogram.

    Args:
        img : grayscale image array (H, W), values 0-255 (uint8)
        bins: number of histogram bins (256 for full range)

    Returns:
        Normalized histogram of shape (bins,) — sums to 1.0
    """
    hist, _ = np.histogram(img.flatten(), bins=bins, range=(0, 255))
    hist = hist.astype(np.float32)

    # Normalize so values sum to 1.0
    total = hist.sum()
    if total > 0:
        hist = hist / total

    return hist


def compute_chi_square_statistic(img: np.ndarray) -> float:
    """
    Chi-square test for detecting LSB steganography.

    THE THEORY (simple version):
      In a natural image, pixel values are smoothly distributed.
      After LSB embedding, values at (0,1), (2,3), (4,5), ... become similar.
      This happens because when you embed a random bit, a pixel that was 200
      gets changed to 201 (or stays 200), and a pixel that was 201 might stay
      201 (or get changed to 200). So adjacent even-odd pairs "merge."

      The chi-square statistic measures how much this pairing deviates from
      the expected distribution. Higher chi-square = more likely stego.

    Returns:
        Chi-square statistic (float). Higher = more suspicious.
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 255))

    # Pair up adjacent bins: (0,1), (2,3), (4,5), ...
    # These pairs should have similar counts in a stego image
    pairs_expected = []
    pairs_observed_0 = []
    pairs_observed_1 = []

    for i in range(0, 256 - 1, 2):
        pair_sum = hist[i] + hist[i + 1]
        expected = pair_sum / 2.0         # if stego: both should be equal
        pairs_expected.append(expected)
        pairs_observed_0.append(hist[i])
        pairs_observed_1.append(hist[i + 1])

    pairs_expected = np.array(pairs_expected, dtype=np.float64)
    pairs_observed_0 = np.array(pairs_observed_0, dtype=np.float64)

    # Chi-square = sum( (observed - expected)^2 / expected )
    mask = pairs_expected > 0
    chi_sq = np.sum(
        (pairs_observed_0[mask] - pairs_expected[mask]) ** 2 / pairs_expected[mask]
    )
    return float(chi_sq)


def compute_entropy(img: np.ndarray) -> float:
    """
    Shannon entropy of the pixel value distribution.
    Stego images tend to have slightly higher entropy than clean ones.
    """
    hist = extract_histogram(img, bins=256)
    # Entropy: -sum(p * log2(p)) where p > 0
    mask = hist > 0
    entropy = -np.sum(hist[mask] * np.log2(hist[mask]))
    return float(entropy)


def compute_local_variance(img: np.ndarray, block_size: int = 16) -> Tuple[float, float]:
    """
    Mean and standard deviation of local variance across the image.
    Computed on non-overlapping blocks of block_size × block_size.

    Steganography can change the local variance pattern slightly.
    """
    H, W = img.shape
    variances = []

    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            block = img[i:i+block_size, j:j+block_size].astype(np.float64)
            variances.append(np.var(block))

    if not variances:
        return 0.0, 0.0

    variances = np.array(variances)
    return float(variances.mean()), float(variances.std())


def extract_statistical_features(img: np.ndarray) -> np.ndarray:
    """
    Extract ALL statistical features from a single grayscale image.

    Args:
        img: numpy array shape (H, W), values 0-255 (uint8 or float)

    Returns:
        Feature vector of length 262:
          - 256 histogram bins (normalized)
          - chi-square statistic (1)
          - entropy (1)
          - mean, std, skewness, kurtosis (4)

    Total: 256 + 1 + 1 + 4 = 262 features
    """
    img_uint8 = img.astype(np.uint8)
    img_float = img.astype(np.float64)

    # Feature 1-256: Histogram (256 bins)
    hist = extract_histogram(img_uint8, bins=256)

    # Feature 257: Chi-square statistic (normalized to reasonable range)
    chi_sq = compute_chi_square_statistic(img_uint8)
    chi_sq_norm = np.log1p(chi_sq) / 10.0    # log-compress + scale

    # Feature 258: Entropy
    entropy = compute_entropy(img_uint8)
    entropy_norm = entropy / 8.0              # max entropy = 8 bits → normalize to [0,1]

    # Features 259-262: Basic statistics
    flat = img_float.flatten()
    mean_val = flat.mean() / 255.0
    std_val = flat.std() / 255.0 * 4.0        # scale std to ~1.0 range
    skew_val = float(scipy_skew(flat))
    kurt_val = float(scipy_kurtosis(flat))

    # Clamp skew and kurtosis to reasonable bounds
    skew_val = np.clip(skew_val, -5.0, 5.0) / 5.0
    kurt_val = np.clip(kurt_val, -5.0, 5.0) / 5.0

    # Concatenate all features
    features = np.concatenate([
        hist,                                                       # 256 bins
        np.array([chi_sq_norm, entropy_norm,
                  mean_val, std_val, skew_val, kurt_val])           # 6 scalars
    ]).astype(np.float32)

    return features   # shape: (262,)


FEATURE_DIM = 262   # total features (update this if you add more)


# ─── MLP Model ────────────────────────────────────────────────────────────────

class StatisticsMLP(nn.Module):
    """
    Branch C: Statistical Features Multi-Layer Perceptron.

    Takes a handcrafted feature vector and outputs a compact feature
    representation for the fusion layer.

    Architecture:
        Feature vector (262-dim)
          ↓
        Linear(262, 256) → BN1d → ReLU → Dropout
          ↓
        Linear(256, 128) → BN1d → ReLU → Dropout
          ↓
        Linear(128, 32) → ReLU
          ↓
        feature_vector (32-dim)  ← sent to fusion layer
    """

    def __init__(self, input_dim: int = FEATURE_DIM, feature_dim: int = 32):
        """
        Args:
            input_dim  : dimension of handcrafted feature vector (262)
            feature_dim: size of output feature vector sent to fusion (32)
        """
        super().__init__()
        self.feature_dim = feature_dim

        self.mlp = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # Layer 3 — output feature vector
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature vector tensor of shape (batch, input_dim)

        Returns:
            compressed feature vector of shape (batch, feature_dim)
        """
        return self.mlp(x)


class BranchCClassifier(nn.Module):
    """
    Standalone Branch C model with classification head.
    Input: RAW IMAGE tensor — statistical features are extracted inside.
    """

    def __init__(self, feature_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.backbone = StatisticsMLP(input_dim=FEATURE_DIM, feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _extract_features_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract statistical features from a batch of image tensors.
        This is CPU-side feature extraction (can't be GPU-accelerated easily).

        Args:
            x: image tensors (batch, 1, H, W), range [-1, 1]

        Returns:
            feature tensor (batch, FEATURE_DIM)
        """
        batch_features = []
        for b in range(x.shape[0]):
            # Convert tensor to numpy uint8 image
            img_np = x[b, 0].cpu().numpy()
            img_np = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            features = extract_statistical_features(img_np)
            batch_features.append(features)

        feat_array = np.stack(batch_features, axis=0)    # (batch, 262)
        return torch.tensor(feat_array, dtype=torch.float32).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw image tensor (batch, 1, H, W)

        Returns:
            logits (batch, 2)
        """
        feat_tensor = self._extract_features_batch(x)
        features = self.backbone(feat_tensor)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion model.
        """
        feat_tensor = self._extract_features_batch(x)
        return self.backbone(feat_tensor)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Branch C (Statistical MLP)...")

    # Test feature extraction on a single dummy image
    dummy_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    features = extract_statistical_features(dummy_img)
    print(f"Feature vector length: {len(features)} (should be {FEATURE_DIM})")
    print(f"Chi-square component: {features[256]:.4f}")
    print(f"Entropy component:    {features[257]:.4f}")

    # Test full model
    batch = torch.randn(4, 1, 256, 256)
    model = BranchCClassifier(feature_dim=32, num_classes=2)
    model.eval()

    with torch.no_grad():
        logits = model(batch)
        branch_features = model.get_features(batch)

    print(f"\nInput shape:    {batch.shape}")
    print(f"Feature shape:  {branch_features.shape}  ← this goes into fusion model")
    print(f"Output shape:   {logits.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}  (this branch is intentionally small)")
    print("\n✅ Branch C OK!")
