# Steganography Detection Using Multi-Branch CNN
### EDI Project | 2nd Year CSE | 3-Month Plan

A parallel three-branch CNN system that detects image steganography (hidden data smuggled inside normal-looking images) across **pixel**, **frequency**, and **statistical** domains simultaneously. Designed for deployment on **air-gapped CPU-only defense infrastructure** (e.g., DRDO endpoints).

---

## What This Project Does

| Branch | What it checks | Catches |
|---|---|---|
| Branch A — Pixel CNN | Raw pixel residual maps via SRM filters | LSB steganography |
| Branch B — DCT CNN | Frequency-domain (DCT) coefficient maps | JPEG-domain stego |
| Branch C — Stats MLP | Histogram, chi-square, entropy features | Classical statistical anomalies |
| Fusion Layer | Combines all three → confidence score | All of the above |

**Extra modules:**
- 🔐 **Defense-Context Scorer** — risk level based on user privilege, time, file size, destination IP
- ⚡ **INT8 Quantization** — model runs on CPU without any GPU

---

## Project Structure

```
steganalysis_project/
├── data/
│   ├── raw/          # Downloaded BOSS dataset images
│   ├── clean/        # 5,000 unmodified clean images
│   └── stego/        # 5,000 LSB-embedded stego images
├── src/
│   ├── data_pipeline/
│   │   ├── lsb_embedder.py    # Creates stego images
│   │   ├── dataset.py         # PyTorch Dataset class
│   │   └── transforms.py      # Image transforms
│   ├── branches/
│   │   ├── branch_a_pixel.py  # SRM + Pixel CNN
│   │   ├── branch_b_dct.py    # DCT + Frequency CNN
│   │   └── branch_c_stats.py  # Stats MLP
│   ├── model/
│   │   ├── fusion_model.py    # Full 3-branch model
│   │   └── srm_filters.py     # SRM filter kernels
│   ├── training/
│   │   ├── train.py           # Training loop
│   │   ├── evaluate.py        # Metrics + plots
│   │   └── config.py          # Hyperparameters
│   ├── deployment/
│   │   ├── quantize.py        # INT8 quantization
│   │   └── inference.py       # CPU inference
│   └── defense_context/
│       └── risk_scorer.py     # Defense risk scoring
├── notebooks/                 # Colab-ready training notebooks
├── app/
│   └── streamlit_demo.py      # Web demo (upload → detect)
├── weights/                   # Saved .pt model files
├── results/                   # Plots, confusion matrices
├── scripts/                   # Dataset setup scripts
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up the dataset
```bash
# Download BOSS dataset (requires registration at agents.cz/boss)
bash scripts/setup_boss_dataset.sh

# Create stego PNG images from clean images
# (the `dataset` subcommand is required; output is always saved as .png to preserve LSB bits)
python src/data_pipeline/lsb_embedder.py dataset --input data/clean --output data/stego
```

### 3. Train the model
```bash
# Train each branch first, then the fusion model
python src/training/train.py --mode branch_a
python src/training/train.py --mode branch_b
python src/training/train.py --mode branch_c
python src/training/train.py --mode fusion
```

### 4. Evaluate
```bash
python src/training/evaluate.py --checkpoint weights/fusion_best.pt
```

### 5. Run the demo
```bash
streamlit run app/streamlit_demo.py
```

---

## Dataset

- **BOSS Dataset** — 10,000 grayscale images (512×512). Register at: http://www.agents.cz/boss/
- **DIV2K** — high-quality colour images for generalisation testing
- We create stego images by running our own LSB embedder on 5,000 images

---

## Tech Stack

Python 3.10 · PyTorch 2.x · OpenCV · scipy · scikit-learn · Streamlit

---

## Team

Student A — data pipeline, Branch A (pixel), quantization  
Student B — training loop, Branch B (DCT), evaluation metrics  
Both — Branch C (stats), fusion layer, Streamlit demo, report

---

## Academic Targets

- **Journal:** IETE Journal of Research (SCOPUS) or IET Image Processing  
- **Backup:** Defence Science Journal (DRDO, free, SCOPUS)  
- **Conference:** NCC (IIT hosted) or INDICON (IEEE India)
- **Patent:** File provisional BEFORE public submission (INR 1,750)
