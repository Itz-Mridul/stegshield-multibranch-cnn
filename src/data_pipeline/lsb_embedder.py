"""
lsb_embedder.py
===============
LSB (Least Significant Bit) Steganography Tool

WHAT THIS DOES:
  - EMBED: Hides secret data inside an image by changing the last bit
           of each pixel value by 0 or 1. Human eye cannot see the difference.
  - EXTRACT: Reads back the hidden data from a stego image.
  - CREATE DATASET: Batch-embeds random data into a folder of clean images.

HOW LSB WORKS:
  Normal pixel: 200 = 11001000 in binary
  Attacker hides bit '1': changes to 201 = 11001001
  Human eye sees ZERO difference. But you can hide megabytes this way.

STUDENT A TASK: Run this to create your stego dataset (Week 2-3).
"""

import os
import random
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ─── Core LSB Functions ──────────────────────────────────────────────────────

def embed_lsb(image: np.ndarray, payload: bytes) -> np.ndarray:
    """
    Embed payload bytes into an image using LSB steganography.

    How it works step by step:
      1. Convert each byte of payload into 8 individual bits.
      2. For each pixel in the image (flattened), take the last bit of the
         pixel value and replace it with the next payload bit.
      3. Rebuild the image from the modified pixels.

    Args:
        image   : numpy array of shape (H, W) for grayscale or (H, W, C) for RGB
        payload : bytes to hide inside the image

    Returns:
        Modified numpy array (same shape) with payload hidden inside.

    Raises:
        ValueError: if payload is too large to fit in this image.
    """
    flat = image.flatten().copy()  # work on a flat copy

    # Convert payload bytes to a list of individual bits
    payload_bits = []
    for byte in payload:
        for bit_pos in range(7, -1, -1):          # MSB first
            payload_bits.append((byte >> bit_pos) & 1)

    # Check capacity: one bit per pixel
    if len(payload_bits) > len(flat):
        raise ValueError(
            f"Payload too large: need {len(payload_bits)} bits, "
            f"but image only has {len(flat)} pixels."
        )

    # Replace LSB of each pixel with one payload bit
    for i, bit in enumerate(payload_bits):
        flat[i] = (flat[i] & 0xFE) | bit   # clear last bit, set to payload bit

    return flat.reshape(image.shape)


def extract_lsb(image: np.ndarray, payload_length_bytes: int) -> bytes:
    """
    Extract hidden payload from a stego image.

    Args:
        image               : stego numpy array
        payload_length_bytes: how many bytes were hidden (must match embed call)

    Returns:
        Extracted bytes payload.
    """
    flat = image.flatten()
    bits_needed = payload_length_bytes * 8

    # Read LSB of first N pixels
    bits = [int(flat[i]) & 1 for i in range(bits_needed)]

    # Group bits back into bytes
    extracted = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        byte_val = 0
        for bit in byte_bits:
            byte_val = (byte_val << 1) | bit
        extracted.append(byte_val)

    return bytes(extracted)


# ─── Dataset Creation ─────────────────────────────────────────────────────────

def create_stego_image(
    input_path: str,
    output_path: str,
    payload_size_bytes: Optional[int] = None,
    seed: Optional[int] = None
) -> int:
    """
    Take a single clean image, embed random payload, save as stego image.

    Args:
        input_path        : path to the clean image (.png / .jpg)
        output_path       : where to save the stego image
        payload_size_bytes: how many bytes to hide (default: 10% of capacity)
        seed              : random seed for reproducibility

    Returns:
        Number of bytes embedded.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Load image as numpy array
    img = Image.open(input_path).convert("L")  # convert to grayscale (L mode)
    img_array = np.array(img, dtype=np.uint8)

    total_pixels = img_array.size

    # Default: hide data using 10% of pixel capacity
    if payload_size_bytes is None:
        payload_size_bytes = total_pixels // (8 * 10)   # 10% capacity

    # Clamp to max capacity
    max_bytes = total_pixels // 8
    payload_size_bytes = min(payload_size_bytes, max_bytes)

    # Generate random payload (simulates a hidden document)
    payload = bytes([random.randint(0, 255) for _ in range(payload_size_bytes)])

    # Embed the payload
    stego_array = embed_lsb(img_array, payload)

    # Save stego image
    stego_img = Image.fromarray(stego_array.astype(np.uint8))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stego_img.save(output_path)

    return payload_size_bytes


def create_stego_dataset(
    clean_dir: str,
    output_dir: str,
    max_images: Optional[int] = None,
    payload_fraction: float = 0.1,
    verbose: bool = True
) -> dict:
    """
    Batch create stego images from a directory of clean images.

    Args:
        clean_dir       : folder with clean PNG/JPG images
        output_dir      : folder to save stego images
        max_images      : limit how many images to process (None = all)
        payload_fraction: fraction of pixel capacity to fill (0.1 = 10%)
        verbose         : print progress

    Returns:
        dict with statistics (num_processed, num_failed, etc.)
    """
    clean_path = Path(clean_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all image files
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
    image_files = [
        f for f in sorted(clean_path.iterdir())
        if f.suffix.lower() in extensions
    ]

    if max_images is not None:
        image_files = image_files[:max_images]

    stats = {"processed": 0, "failed": 0, "total": len(image_files)}

    for idx, img_file in enumerate(image_files):
        out_file = output_path / img_file.name
        try:
            bytes_embedded = create_stego_image(
                input_path=str(img_file),
                output_path=str(out_file),
                seed=idx  # deterministic per image
            )
            stats["processed"] += 1

            if verbose and (idx + 1) % 100 == 0:
                print(f"[{idx + 1}/{len(image_files)}] Done: {img_file.name} "
                      f"({bytes_embedded} bytes hidden)")

        except Exception as e:
            stats["failed"] += 1
            if verbose:
                print(f"[FAILED] {img_file.name}: {e}")

    if verbose:
        print(f"\n✅ Done! Processed: {stats['processed']}, "
              f"Failed: {stats['failed']}, Total: {stats['total']}")

    return stats


# ─── Verification ─────────────────────────────────────────────────────────────

def verify_embedding(clean_path: str, stego_path: str, payload_size_bytes: int) -> bool:
    """
    Sanity check: verify that hidden data can be correctly extracted.
    Also confirms the PSNR (image quality loss) is negligible.
    """
    clean_img = np.array(Image.open(clean_path).convert("L"), dtype=np.uint8)
    stego_img = np.array(Image.open(stego_path).convert("L"), dtype=np.uint8)

    # Check PSNR — should be > 50 dB (imperceptible to human eye)
    mse = np.mean((clean_img.astype(float) - stego_img.astype(float)) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)

    print(f"  PSNR: {psnr:.2f} dB  (should be > 50 dB)")
    print(f"  Max pixel difference: {np.max(np.abs(clean_img.astype(int) - stego_img.astype(int)))}"
          f"  (should be 0 or 1)")

    return psnr > 50.0


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LSB Steganography Tool — embeds and extracts hidden data in images."
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── embed ──
    embed_parser = subparsers.add_parser("embed", help="Embed data into a single image")
    embed_parser.add_argument("--input", required=True, help="Input clean image path")
    embed_parser.add_argument("--output", required=True, help="Output stego image path")
    embed_parser.add_argument("--bytes", type=int, default=None, help="Payload size in bytes")

    # ── dataset ──
    dataset_parser = subparsers.add_parser("dataset", help="Batch create stego dataset")
    dataset_parser.add_argument("--input", required=True, help="Folder with clean images")
    dataset_parser.add_argument("--output", required=True, help="Folder to save stego images")
    dataset_parser.add_argument("--max", type=int, default=None, help="Max images to process")

    # ── verify ──
    verify_parser = subparsers.add_parser("verify", help="Verify a stego image")
    verify_parser.add_argument("--clean", required=True, help="Original clean image")
    verify_parser.add_argument("--stego", required=True, help="Stego image to verify")
    verify_parser.add_argument("--bytes", type=int, default=100, help="Bytes that were embedded")

    # ── test ──
    subparsers.add_parser("test", help="Run self-test with a dummy image")

    args = parser.parse_args()

    if args.command == "embed":
        n = create_stego_image(args.input, args.output, args.bytes)
        print(f"✅ Embedded {n} bytes into {args.output}")

    elif args.command == "dataset":
        create_stego_dataset(args.input, args.output, max_images=args.max)

    elif args.command == "verify":
        ok = verify_embedding(args.clean, args.stego, args.bytes)
        print("✅ Embedding verified!" if ok else "❌ Verification failed!")

    elif args.command == "test":
        print("Running self-test...")
        # Create a dummy 100×100 grayscale image
        dummy = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        payload = b"Hello DRDO! This is a secret message hidden in pixels."

        stego = embed_lsb(dummy, payload)
        recovered = extract_lsb(stego, len(payload))

        assert recovered == payload, "❌ Self-test FAILED — extracted data doesn't match!"
        print(f"✅ Self-test PASSED! Embedded and extracted {len(payload)} bytes correctly.")
        print(f"   Max pixel difference: {np.max(np.abs(dummy.astype(int) - stego.astype(int)))}")
        print(f"   Original: {dummy[0, :5]}")
        print(f"   Stego:    {stego[0, :5]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
