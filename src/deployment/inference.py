"""
inference.py
============
CPU Inference Wrapper

WHAT THIS DOES:
  Clean, simple wrapper around the trained (or quantized) model
  for running predictions on individual images or folders.

  Designed for CPU-only deployment on air-gapped machines.

HOW TO USE:
  # Predict a single image:
  python src/deployment/inference.py --image path/to/image.png

  # Predict all images in a folder:
  python src/deployment/inference.py --folder path/to/images/ --output results.csv

  # Use INT8 quantized model (faster on CPU):
  python src/deployment/inference.py --image test.png --int8 --checkpoint weights/fusion_int8.pt
"""

import os
import sys
import argparse
import time
import csv
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from training.config import Config
from model.fusion_model import MultiBranchSteganalyzer


# ─── Predictor Class ──────────────────────────────────────────────────────────

class SteganalysiPredictor:
    """
    Production-ready inference class for steganalysis.

    Usage:
        predictor = SteganalysiPredictor(checkpoint="weights/fusion_best.pt")
        result = predictor.predict("suspicious_image.png")
        print(result)
        # {'label': 'STEGO', 'confidence': 0.947, 'stego_prob': 0.947}
    """

    def __init__(
        self,
        checkpoint: str = "weights/fusion_best.pt",
        use_int8: bool = False,
        image_size: int = 256,
        cfg: Optional[Config] = None
    ):
        """
        Args:
            checkpoint : path to trained model weights (.pt file)
            use_int8   : load as INT8 quantized model (faster on CPU)
            image_size : resize images to this square before inference
            cfg        : Config object (uses defaults if None)
        """
        self.cfg        = cfg or Config()
        self.image_size = image_size
        self.device     = torch.device("cpu")   # always CPU for deployment
        self.use_int8   = use_int8

        # Load model
        print(f"Loading model from: {checkpoint}")
        self.model = self._load_model(checkpoint, use_int8)
        self.model.eval()
        print("Model ready for inference.")

    def _load_model(self, checkpoint: str, use_int8: bool) -> torch.nn.Module:
        """Load and return the model."""
        model = MultiBranchSteganalyzer(
            feature_dim_a=self.cfg.feature_dim_a,
            feature_dim_b=self.cfg.feature_dim_b,
            feature_dim_c=self.cfg.feature_dim_c
        )

        if use_int8:
            import torch.quantization
            model = torch.quantization.quantize_dynamic(
                model.cpu(),
                qconfig_spec={torch.nn.Linear},
                dtype=torch.qint8
            )

        if os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}\n"
                "Train the model first:\n"
                "  python src/training/train.py --mode branch_a\n"
                "  python src/training/train.py --mode branch_b\n"
                "  python src/training/train.py --mode branch_c\n"
                "  python src/training/train.py --mode fusion\n"
                "Or use --checkpoint to specify a different path."
            )

        return model

    def _preprocess(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Load and preprocess an image into a model-ready tensor.

        Accepts:
          - str: file path
          - numpy array: (H, W) uint8
          - PIL Image

        Returns:
            tensor of shape (1, 1, image_size, image_size)
        """
        if isinstance(image, str):
            img = Image.open(image).convert("L")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("L")
        elif isinstance(image, Image.Image):
            img = image.convert("L")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Use centre crop — NOT resize — to preserve exact LSB pixel values.
        # Bilinear interpolation averages neighbours, which corrupts 1-bit LSB payloads.
        W, H = img.size
        if W < self.image_size or H < self.image_size:
            raise ValueError(
                f"Image {W}×{H} is smaller than crop_size={self.image_size}. "
                "Either use a larger image or reduce crop_size in the Config."
            )
        # Centre crop
        left = (W - self.image_size) // 2
        top  = (H - self.image_size) // 2
        img  = img.crop((left, top, left + self.image_size, top + self.image_size))
        arr  = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5                           # normalize to [-1, 1]
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> dict:
        """
        Run inference on a single image.

        Args:
            image: file path, numpy array, or PIL Image

        Returns:
            dict with:
              - 'label':      'CLEAN' or 'STEGO'
              - 'confidence': float 0.0-1.0 (probability of predicted class)
              - 'stego_prob': float 0.0-1.0 (always the stego probability)
              - 'clean_prob': float 0.0-1.0
              - 'inference_ms': time taken in milliseconds
        """
        tensor = self._preprocess(image)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        clean_prob = probs[0, 0].item()
        stego_prob = probs[0, 1].item()
        predicted  = 1 if stego_prob >= 0.5 else 0
        confidence = stego_prob if predicted == 1 else clean_prob

        return {
            "label":         "STEGO" if predicted == 1 else "CLEAN",
            "predicted_class": predicted,
            "confidence":    round(confidence, 4),
            "stego_prob":    round(stego_prob, 4),
            "clean_prob":    round(clean_prob, 4),
            "inference_ms":  round(elapsed_ms, 2)
        }

    def predict_folder(self, folder: str) -> list:
        """
        Run inference on all images in a folder.

        Args:
            folder: path to folder containing images

        Returns:
            list of dicts, one per image, with filename + prediction
        """
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
        folder_path = Path(folder)
        image_files = sorted([
            f for f in folder_path.iterdir()
            if f.suffix.lower() in extensions
        ])

        print(f"Processing {len(image_files)} images from {folder}...")
        results = []

        for idx, img_file in enumerate(image_files, 1):
            result = self.predict(str(img_file))
            result["filename"] = img_file.name
            results.append(result)

            if idx % 50 == 0:
                print(f"  [{idx}/{len(image_files)}] Done")

        stego_count = sum(1 for r in results if r["predicted_class"] == 1)
        print(f"\nDone! {stego_count}/{len(results)} images flagged as STEGO.")
        return results

    def benchmark(self, n_runs: int = 100) -> dict:
        """
        Measure average inference speed.
        Useful for your paper's deployment section (Week 11).
        """
        dummy = torch.randn(1, 1, self.image_size, self.image_size)

        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy)

        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "avg_ms":    round(np.mean(times), 2),
            "min_ms":    round(np.min(times), 2),
            "max_ms":    round(np.max(times), 2),
            "std_ms":    round(np.std(times), 2),
            "throughput_fps": round(1000 / np.mean(times), 1)
        }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run steganalysis inference on images")
    parser.add_argument("--image",      type=str, help="Single image path")
    parser.add_argument("--folder",     type=str, help="Folder of images")
    parser.add_argument("--output",     type=str, help="Save results to CSV file")
    parser.add_argument("--checkpoint", type=str, default="weights/fusion_best.pt")
    parser.add_argument("--int8",       action="store_true", help="Use INT8 quantized model")
    parser.add_argument("--benchmark",  action="store_true", help="Run speed benchmark")
    args = parser.parse_args()

    predictor = SteganalysiPredictor(
        checkpoint=args.checkpoint,
        use_int8=args.int8
    )

    if args.benchmark:
        print("\nRunning speed benchmark (100 runs)...")
        stats = predictor.benchmark()
        print(f"\n  Average inference time: {stats['avg_ms']} ms/image")
        print(f"  Min: {stats['min_ms']} ms  |  Max: {stats['max_ms']} ms")
        print(f"  Throughput: {stats['throughput_fps']} images/second (CPU)")
        print("\n  → Copy these numbers into your paper's Table of Results!")

    elif args.image:
        result = predictor.predict(args.image)
        print(f"\n  File:        {args.image}")
        print(f"  Prediction:  {result['label']}")
        print(f"  Confidence:  {result['confidence']*100:.1f}%")
        print(f"  Stego prob:  {result['stego_prob']*100:.1f}%")
        print(f"  Time:        {result['inference_ms']} ms")

    elif args.folder:
        results = predictor.predict_folder(args.folder)

        if args.output:
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"✅ Results saved to: {args.output}")
        else:
            for r in results[:20]:   # print first 20
                print(f"  {r['filename']:<30} {r['label']:<8} "
                      f"({r['confidence']*100:.1f}%) {r['inference_ms']}ms")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
