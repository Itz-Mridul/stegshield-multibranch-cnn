"""
fusion_model.py
===============
Full Three-Branch Fusion Model

WHAT THIS DOES:
  Combines Branch A (pixel), Branch B (DCT frequency), and Branch C (statistics)
  into one complete steganalysis system.

  Key insight: Each branch outputs a FEATURE VECTOR, not a class prediction.
  The fusion layer takes all three feature vectors, concatenates them,
  and learns the optimal way to combine them.

  CRITICAL STEP: L2-normalization before concatenation.
    Without this, one branch (usually Branch A with 256 dims) dominates
    over Branch C (32 dims), and the fusion training fails to converge.
    Each branch gets normalized independently first.

FUSION ARCHITECTURE:
    Branch A features (256-dim) ─┐
    Branch B features (256-dim) ─┼─ L2 Norm ─ Concat (544-dim)
    Branch C features  (32-dim) ─┘
                                     ↓
                              Dense(544 → 128) → ReLU → Dropout(0.3)
                                     ↓
                              Dense(128 → 64) → ReLU → Dropout(0.2)
                                     ↓
                              Dense(64 → 2) → Softmax
                                     ↓
                        [CLEAN / STEGO + confidence %]

JOINT FINE-TUNING (Weeks 9-10):
  All branches are trained together in the fusion model.
  This allows the branches to specialise for the combined task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from branches.branch_a_pixel import PixelCNN
from branches.branch_b_dct import FrequencyCNN, image_to_dct_tensor
from branches.branch_c_stats import StatisticsMLP, extract_statistical_features, FEATURE_DIM
import numpy as np


class FusionLayer(nn.Module):
    """
    The 'Senior Inspector' that combines all three branch outputs.

    Input:  concatenated feature vector from A + B + C
    Output: (batch, 2) logits → class probabilities via softmax
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        """
        Args:
            input_dim  : total size of concatenated feature vector (A+B+C)
            num_classes: 2 (clean vs stego)
        """
        super().__init__()

        self.net = nn.Sequential(
            # First dense layer — large to small compression
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # Second dense layer
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # Output layer
            nn.Linear(64, num_classes)
            # Note: NO softmax here — use CrossEntropyLoss which includes softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: concatenated features of shape (batch, input_dim)

        Returns:
            logits of shape (batch, num_classes)
        """
        return self.net(x)


class MultiBranchSteganalyzer(nn.Module):
    """
    Complete three-branch fusion model for steganography detection.

    This is the FINAL MODEL that combines everything.

    Usage:
        model = MultiBranchSteganalyzer()
        logits = model(images)                     # inference
        probs = F.softmax(logits, dim=1)           # probabilities
        confidence, predicted_class = probs.max(1) # confidence + class
    """

    def __init__(
        self,
        feature_dim_a: int = 256,
        feature_dim_b: int = 256,
        feature_dim_c: int = 32,
        num_classes: int = 2
    ):
        """
        Args:
            feature_dim_a : Branch A output size (pixel CNN)
            feature_dim_b : Branch B output size (DCT CNN)
            feature_dim_c : Branch C output size (stats MLP)
            num_classes   : 2 = clean/stego
        """
        super().__init__()

        # ── Three Branches ────────────────────────────────────────────────────
        self.branch_a = PixelCNN(feature_dim=feature_dim_a)
        self.branch_b = FrequencyCNN(feature_dim=feature_dim_b)
        self.branch_c = StatisticsMLP(input_dim=FEATURE_DIM, feature_dim=feature_dim_c)

        # ── Fusion Layer ──────────────────────────────────────────────────────
        fusion_input = feature_dim_a + feature_dim_b + feature_dim_c
        self.fusion = FusionLayer(input_dim=fusion_input, num_classes=num_classes)

        # Store dims for reference
        self.feature_dim_a = feature_dim_a
        self.feature_dim_b = feature_dim_b
        self.feature_dim_c = feature_dim_c

    def _get_branch_c_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract statistical features from image tensor for Branch C.
        This involves CPU-side numpy computation.
        """
        batch_features = []
        for b in range(x.shape[0]):
            img_np = x[b, 0].cpu().numpy()
            img_np = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            features = extract_statistical_features(img_np)
            batch_features.append(features)

        feat_array = np.stack(batch_features, axis=0)
        return torch.tensor(feat_array, dtype=torch.float32).to(x.device)

    def forward(
        self,
        x: torch.Tensor,
        return_branch_features: bool = False
    ):
        """
        Forward pass through the full three-branch fusion model.

        Args:
            x                    : grayscale image tensor (batch, 1, H, W)
            return_branch_features: if True, also returns individual branch features

        Returns:
            logits (batch, 2)
            OR (logits, feat_a, feat_b, feat_c) if return_branch_features=True
        """
        # ── Branch A: Pixel features ──────────────────────────────────────────
        feat_a = self.branch_a(x)                       # (batch, 256)

        # ── Branch B: DCT frequency features ─────────────────────────────────
        dct_map = image_to_dct_tensor(x).to(x.device)
        feat_b = self.branch_b(dct_map)                  # (batch, 256)

        # ── Branch C: Statistical features ───────────────────────────────────
        stat_tensor = self._get_branch_c_features(x)    # (batch, 262)
        feat_c = self.branch_c(stat_tensor)              # (batch, 32)

        # ── L2 Normalization (CRITICAL!) ──────────────────────────────────────
        # Without this, Branch A (256-dim) will dominate Branch C (32-dim)
        # and fusion training may never converge.
        feat_a_norm = F.normalize(feat_a, p=2, dim=1)   # (batch, 256)
        feat_b_norm = F.normalize(feat_b, p=2, dim=1)   # (batch, 256)
        feat_c_norm = F.normalize(feat_c, p=2, dim=1)   # (batch, 32)

        # ── Concatenate ───────────────────────────────────────────────────────
        combined = torch.cat([feat_a_norm, feat_b_norm, feat_c_norm], dim=1)
        # Shape: (batch, 256+256+32) = (batch, 544)

        # ── Fusion Layer → Final Prediction ──────────────────────────────────
        logits = self.fusion(combined)                   # (batch, 2)

        if return_branch_features:
            return logits, feat_a, feat_b, feat_c

        return logits

    def predict(self, x: torch.Tensor) -> dict:
        """
        User-friendly inference method.

        Returns:
            dict with keys:
              - 'class':      0 (clean) or 1 (stego) per image
              - 'confidence': probability of predicted class (0.0 to 1.0)
              - 'stego_prob': probability of being stego (for ROC curves)
              - 'label':      "CLEAN" or "STEGO" string
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)              # (batch, 2)
            stego_prob = probs[:, 1]                      # P(stego)
            confidence, predicted = probs.max(dim=1)     # max prob + class

        labels = ["CLEAN" if c == 0 else "STEGO" for c in predicted.cpu().numpy()]

        return {
            "class":       predicted.cpu().numpy(),
            "confidence":  confidence.cpu().numpy(),
            "stego_prob":  stego_prob.cpu().numpy(),
            "label":       labels
        }

    def load_pretrained_branches(
        self,
        branch_a_path: str = None,
        branch_b_path: str = None,
        branch_c_path: str = None
    ):
        """
        Load individually trained branch weights into the fusion model.
        Call this before joint fine-tuning to start from good branch weights.

        Args:
            branch_a_path: path to saved Branch A .pt file
            branch_b_path: path to saved Branch B .pt file
            branch_c_path: path to saved Branch C .pt file
        """
        if branch_a_path and os.path.exists(branch_a_path):
            state = torch.load(branch_a_path, map_location="cpu")
            # Branch A was saved as BranchAClassifier — extract backbone weights
            backbone_state = {
                k.replace("backbone.", ""): v
                for k, v in state.items()
                if k.startswith("backbone.")
            }
            self.branch_a.load_state_dict(backbone_state, strict=False)
            print(f"✅ Branch A weights loaded from {branch_a_path}")

        if branch_b_path and os.path.exists(branch_b_path):
            state = torch.load(branch_b_path, map_location="cpu")
            backbone_state = {
                k.replace("backbone.", ""): v
                for k, v in state.items()
                if k.startswith("backbone.")
            }
            self.branch_b.load_state_dict(backbone_state, strict=False)
            print(f"✅ Branch B weights loaded from {branch_b_path}")

        if branch_c_path and os.path.exists(branch_c_path):
            state = torch.load(branch_c_path, map_location="cpu")
            backbone_state = {
                k.replace("backbone.", ""): v
                for k, v in state.items()
                if k.startswith("backbone.")
            }
            self.branch_c.load_state_dict(backbone_state, strict=False)
            print(f"✅ Branch C weights loaded from {branch_c_path}")


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Full Fusion Model...")
    print("(Branch C feature extraction takes ~5 seconds for a small batch — normal)")

    batch = torch.randn(2, 1, 256, 256)

    model = MultiBranchSteganalyzer(
        feature_dim_a=256,
        feature_dim_b=256,
        feature_dim_c=32
    )
    model.eval()

    # Test forward pass
    with torch.no_grad():
        logits, fa, fb, fc = model(batch, return_branch_features=True)

    print(f"\nInput shape:     {batch.shape}")
    print(f"Branch A output: {fa.shape}   (pixel features)")
    print(f"Branch B output: {fb.shape}   (DCT features)")
    print(f"Branch C output: {fc.shape}    (stat features)")
    print(f"Fusion input:    {fa.shape[1]+fb.shape[1]+fc.shape[1]}-dim concatenated vector")
    print(f"Final output:    {logits.shape}     (batch, 2 classes)")

    # Test predict()
    result = model.predict(batch)
    for i in range(batch.shape[0]):
        print(f"\n  Image {i}: {result['label'][i]} "
              f"(confidence: {result['confidence'][i]*100:.1f}%, "
              f"stego_prob: {result['stego_prob'][i]*100:.1f}%)")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print("\n✅ Full fusion model OK!")
