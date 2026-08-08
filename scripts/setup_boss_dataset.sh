#!/bin/bash
# =============================================================
# setup_boss_dataset.sh
# BOSS Dataset Setup Script
# =============================================================
#
# The BOSS (Break Our Steganographic System) dataset is the
# standard benchmark for all steganalysis research.
#
# HOW TO GET THE DATASET:
#   1. Register at: http://www.agents.cz/boss/
#   2. Download the grayscale base images (BOSSbase_1.01.zip)
#   3. Place the downloaded ZIP in the data/raw/ folder
#   4. Run this script: bash scripts/setup_boss_dataset.sh
#
# ALTERNATIVE (if BOSS registration is slow):
#   Use USC-SIPI image database (public, no registration):
#   https://sipi.usc.edu/database/
#   Or use DIV2K dataset (free download):
#   https://data.vision.ee.ethz.ch/cvl/DIV2K/
#
# WHAT THIS SCRIPT DOES:
#   1. Extracts and organises the BOSS images into data/raw/
#   2. Copies 5,000 images to data/clean/ (unmodified)
#   3. Runs lsb_embedder.py on 5,000 images → data/stego/
#   4. Verifies the dataset integrity
# =============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw"
CLEAN_DIR="$PROJECT_ROOT/data/clean"
STEGO_DIR="$PROJECT_ROOT/data/stego"
ZIP_PATH="$RAW_DIR/BOSSbase_1.01.zip"

echo "======================================================"
echo "  BOSS Dataset Setup"
echo "======================================================"
echo "  Project root: $PROJECT_ROOT"
echo ""

# Create directories
mkdir -p "$RAW_DIR" "$CLEAN_DIR" "$STEGO_DIR"

# ── Step 1: Extract ZIP if it exists ──────────────────────────────────────────
if [ -f "$ZIP_PATH" ]; then
    echo "📦 Extracting BOSS dataset ZIP..."
    unzip -q "$ZIP_PATH" -d "$RAW_DIR"
    echo "✅ Extraction complete."
else
    echo "⚠️  ZIP not found at: $ZIP_PATH"
    echo ""
    echo "   Please download the BOSS dataset:"
    echo "   → http://www.agents.cz/boss/"
    echo "   → File: BOSSbase_1.01.zip"
    echo "   → Place in: data/raw/"
    echo ""
    echo "   ALTERNATIVE: Use USC-SIPI or DIV2K datasets (see comments above)"
    echo ""

    # If no BOSS dataset, create a small synthetic dataset for testing
    echo "🔧 Creating synthetic test dataset (200 images for code verification)..."
    python3 - <<'EOF'
import os
import numpy as np
from PIL import Image

raw_dir  = "data/raw"
os.makedirs(raw_dir, exist_ok=True)

print("Generating 200 synthetic grayscale images (256x256)...")
for i in range(200):
    # Create a synthetic image with natural-looking texture
    noise = np.random.randint(100, 200, (256, 256), dtype=np.uint8)
    # Add some structure to make it look like a natural image
    for r in range(0, 256, 32):
        for c in range(0, 256, 32):
            base = np.random.randint(80, 220)
            noise[r:r+32, c:c+32] = np.clip(
                noise[r:r+32, c:c+32] + base - 150, 0, 255
            ).astype(np.uint8)

    img = Image.fromarray(noise)
    img.save(os.path.join(raw_dir, f"synthetic_{i:04d}.png"))

print(f"✅ Created 200 synthetic images in {raw_dir}/")
print("   Note: Use real BOSS dataset for proper research results!")
EOF
fi

# ── Step 2: Copy clean images ─────────────────────────────────────────────────
echo ""
echo "📋 Copying images to data/clean/..."
find "$RAW_DIR" -name "*.png" -o -name "*.pgm" | head -5000 | while read f; do
    cp "$f" "$CLEAN_DIR/"
done

CLEAN_COUNT=$(ls "$CLEAN_DIR" | wc -l | tr -d ' ')
echo "✅ $CLEAN_COUNT images in data/clean/"

# ── Step 3: Create stego images ───────────────────────────────────────────────
echo ""
echo "🔐 Creating stego images via LSB embedding..."
echo "   This embeds random data into each image."
echo "   This step may take a few minutes for 5,000 images."
echo ""

python3 "$PROJECT_ROOT/src/data_pipeline/lsb_embedder.py" dataset \
    --input "$CLEAN_DIR" \
    --output "$STEGO_DIR"

STEGO_COUNT=$(ls "$STEGO_DIR" | wc -l | tr -d ' ')
echo ""
echo "✅ $STEGO_COUNT stego images created in data/stego/"

# ── Step 4: Verify ────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Dataset Summary"
echo "======================================================"
echo "  Clean images: $(ls "$CLEAN_DIR" | wc -l | tr -d ' ')"
echo "  Stego images: $(ls "$STEGO_DIR" | wc -l | tr -d ' ')"
echo ""

# Verify one pair
FIRST_CLEAN=$(ls "$CLEAN_DIR" | head -1)
if [ -n "$FIRST_CLEAN" ] && [ -f "$STEGO_DIR/$FIRST_CLEAN" ]; then
    echo "🔍 Verifying first image pair..."
    python3 "$PROJECT_ROOT/src/data_pipeline/lsb_embedder.py" verify \
        --clean "$CLEAN_DIR/$FIRST_CLEAN" \
        --stego "$STEGO_DIR/$FIRST_CLEAN" \
        --bytes 100
fi

echo ""
echo "======================================================"
echo "  ✅ Dataset setup complete!"
echo "======================================================"
echo ""
echo "  Next steps:"
echo "  1. Install dependencies:  pip install -r requirements.txt"
echo "  2. Train Branch A:        python src/training/train.py --mode branch_a"
echo "  3. Train Branch B:        python src/training/train.py --mode branch_b"
echo "  4. Train Branch C:        python src/training/train.py --mode branch_c"
echo "  5. Train fusion model:    python src/training/train.py --mode fusion"
echo "  6. Evaluate:              python src/training/evaluate.py --ablation"
echo "  7. Run demo:              streamlit run app/streamlit_demo.py"
echo ""
echo "  📌 FOR GOOGLE COLAB (GPU training):"
echo "     Upload this project folder to Google Drive"
echo "     Open notebooks/05_fusion_training.ipynb in Colab"
echo "     It will mount Drive and train using T4 GPU"
echo ""
