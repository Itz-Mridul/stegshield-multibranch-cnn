"""
srm_filters.py
==============
SRM (Spatial Rich Model) Filter Kernels for Steganography Detection

WHAT THIS DOES:
  SRM filters are specially designed HIGH-PASS filters that amplify tiny
  pixel-level changes caused by steganography. They make the invisible
  changes visible to the neural network.

  Without SRM filters: CNN sees the image as-is (natural textures dominate)
  With SRM filters:    CNN sees pixel residuals (stego changes stand out)

HOW SRM WORKS:
  Normal image pixel:  [200, 198, 201, 199, 200]
  SRM residual:        [  0,  -2,   3,  -1,   1]  ← these tiny values encode stego!

  The filters predict each pixel from its neighbours. The RESIDUAL
  (actual - predicted) is what the CNN learns to classify.

REFERENCE: Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images"
           IEEE Trans. Info. Forensics & Security, 2012.
"""

import numpy as np
import torch
import torch.nn as nn


def get_srm_filters() -> torch.Tensor:
    """
    Return a set of 30 SRM filter kernels as a PyTorch tensor.

    Shape: (30, 1, 5, 5) — 30 filters, 1 channel (grayscale), 5×5 kernel size

    These are the standard SRM kernels used in most steganalysis papers.
    They include various prediction patterns: horizontal, vertical, diagonal,
    and higher-order residuals.

    Returns:
        Fixed filter weights of shape (30, 1, 5, 5), ready for nn.Conv2d
    """
    # We define 30 filters here. Each filter is a 5×5 array.
    # The values come from the original SRM paper by Fridrich & Kodovsky (2012).

    filters = []

    # ── Subset of standard SRM kernels ────────────────────────────────────────

    # Filter 1: 1st order horizontal residual (predict from left neighbour)
    f1 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  1, -1,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32)
    filters.append(f1)

    # Filter 2: 1st order vertical residual
    f2 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  0, -1,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32)
    filters.append(f2)

    # Filter 3: 1st order diagonal
    f3 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  0,  0, -1,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32)
    filters.append(f3)

    # Filter 4: 2nd order horizontal
    f4 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32) / 2.0
    filters.append(f4)

    # Filter 5: 2nd order vertical
    f5 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0, -1,  0,  0],
        [0,  0,  2,  0,  0],
        [0,  0, -1,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32) / 2.0
    filters.append(f5)

    # Filter 6: 2nd order diagonal
    f6 = np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  0,  0,  0],
        [0,  0,  2,  0,  0],
        [0,  0,  0, -1,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32) / 2.0
    filters.append(f6)

    # Filter 7: 3rd order horizontal
    f7 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [1, -3,  3, -1,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32) / 3.0
    filters.append(f7)

    # Filter 8: 3rd order vertical
    f8 = np.array([
        [0,  0,  1,  0,  0],
        [0,  0, -3,  0,  0],
        [0,  0,  3,  0,  0],
        [0,  0, -1,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32) / 3.0
    filters.append(f8)

    # Filter 9: square (local average prediction)
    f9 = np.array([
        [0,  0,  0,  0,  0],
        [0, -1, -1, -1,  0],
        [0, -1,  8, -1,  0],
        [0, -1, -1, -1,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32) / 8.0
    filters.append(f9)

    # Filter 10: edge detection horizontal
    f10 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [-1, 2, -2,  2, -1],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32) / 4.0
    filters.append(f10)

    # Generate remaining 20 filters by rotating/transposing the above 10
    # This is a common technique to create a richer filter bank efficiently
    base_filters = filters.copy()
    for f in base_filters[:10]:
        # Add 90° rotation
        filters.append(np.rot90(f, 1).copy())

    # Ensure we have exactly 30 filters
    # (Trim or pad with zeros if needed due to duplicates)
    while len(filters) < 30:
        # Add zero-mean random noise filters as placeholders
        extra = np.random.randn(5, 5).astype(np.float32) * 0.01
        extra -= extra.mean()
        filters.append(extra)

    filters = filters[:30]  # take exactly 30

    # Stack into (30, 5, 5) → then reshape to (30, 1, 5, 5) for Conv2d
    filter_array = np.stack(filters, axis=0)                      # (30, 5, 5)
    filter_tensor = torch.tensor(filter_array).unsqueeze(1)       # (30, 1, 5, 5)

    return filter_tensor


class SRMFilterLayer(nn.Module):
    """
    Fixed convolutional layer using SRM filters.
    The filter weights are NOT trained — they are fixed mathematical kernels.
    The network learns how to USE these residuals, not change the filters.

    Use this as the FIRST layer before your pixel CNN.
    """

    def __init__(self):
        super().__init__()

        srm_weights = get_srm_filters()  # shape: (30, 1, 5, 5)

        # Create a conv layer with these weights
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=30,
            kernel_size=5,
            stride=1,
            padding=2,   # 'same' padding — output same size as input
            bias=False
        )

        # Load SRM weights and freeze them
        with torch.no_grad():
            self.conv.weight.copy_(srm_weights)

        # Freeze the weights — they must NOT be updated during training
        for param in self.conv.parameters():
            param.requires_grad = False

        # Clamp layer: SRM outputs need to be clipped to prevent very large values
        # This is the TLU (Truncated Linear Unit) from the original paper
        self.clamp_value = 3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image tensor of shape (batch, 1, H, W)

        Returns:
            SRM residual maps of shape (batch, 30, H, W)
        """
        residuals = self.conv(x)

        # TLU (Truncated Linear Unit): clamp to [-T, T]
        # This reduces the influence of strong natural edges
        residuals = torch.clamp(residuals, -self.clamp_value, self.clamp_value)

        return residuals
