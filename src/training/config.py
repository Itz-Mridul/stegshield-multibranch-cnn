"""
config.py
=========
Training Configuration — All hyperparameters in one place.

Change values here instead of hunting through training code.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    clean_dir: str        = "data/clean"
    stego_dir: str        = "data/stego"
    image_size: int       = 256        # resize all images to this square
    max_images: Optional[int] = None   # None = use all images; set 2000 for quick tests
    val_split: float      = 0.20       # 20% validation

    # ── Batch / Workers ───────────────────────────────────────────────────────
    batch_size: int       = 32
    num_workers: int      = 4          # set to 0 on Windows if you hit errors

    # ── Model ─────────────────────────────────────────────────────────────────
    feature_dim_a: int    = 256        # Branch A output feature size
    feature_dim_b: int    = 256        # Branch B output feature size
    feature_dim_c: int    = 32         # Branch C output feature size (smaller)
    num_classes: int      = 2          # 0=clean, 1=stego

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int           = 50
    learning_rate: float  = 1e-4
    weight_decay: float   = 1e-5
    early_stop_patience: int = 8       # stop if val loss doesn't improve for N epochs

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer: str        = "adamw"    # "adamw" or "sgd"
    scheduler: str        = "cosine"   # "cosine" or "step" or "none"
    warmup_epochs: int    = 3          # learning rate warmup

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str   = "weights"
    results_dir: str      = "results"
    best_model_name: str  = "fusion_best.pt"
    branch_a_name: str    = "branch_a_best.pt"
    branch_b_name: str    = "branch_b_best.pt"
    branch_c_name: str    = "branch_c_best.pt"

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int             = 42

    # ── Training Mode ─────────────────────────────────────────────────────────
    # "branch_a"  : train Branch A alone
    # "branch_b"  : train Branch B alone
    # "branch_c"  : train Branch C alone
    # "fusion"    : train full fusion model (needs branches trained first)
    # "joint"     : train all branches + fusion together end-to-end
    mode: str             = "fusion"

    # ── Colab / GPU ───────────────────────────────────────────────────────────
    use_gpu: bool         = True       # auto-detects if GPU available
    mixed_precision: bool = True       # use fp16 on GPU for speed (disable on CPU)

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    @property
    def device(self):
        import torch
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def fusion_input_dim(self) -> int:
        """Total feature vector size entering the fusion layer."""
        return self.feature_dim_a + self.feature_dim_b + self.feature_dim_c

    def summary(self):
        print("=" * 50)
        print("Training Configuration")
        print("=" * 50)
        print(f"Mode:           {self.mode}")
        print(f"Device:         {self.device}")
        print(f"Image size:     {self.image_size}×{self.image_size}")
        print(f"Batch size:     {self.batch_size}")
        print(f"Epochs:         {self.epochs}")
        print(f"Learning rate:  {self.learning_rate}")
        print(f"Max images:     {self.max_images or 'All'}")
        print(f"Fusion input:   {self.fusion_input_dim}-dim vector")
        print("=" * 50)


# Default config — import this in train.py
cfg = Config()
