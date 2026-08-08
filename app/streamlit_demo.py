"""
streamlit_demo.py
=================
Steganography Detection — Interactive Web Demo

HOW TO RUN:
  streamlit run app/streamlit_demo.py

WHAT IT DOES:
  - Upload any image (PNG, JPG, BMP)
  - Model analyzes it through all three branches
  - Shows CLEAN / STEGO verdict + confidence %
  - Shows defense-context risk panel (optional metadata input)
  - Professional UI suitable for professor/jury demo

REQUIREMENT:
  Model checkpoint must exist at: weights/fusion_best.pt
  If not trained yet, use: --test flag to load random weights for demo.
"""

import os
import sys
import io
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(APP_DIR)
SRC_DIR    = os.path.join(PROJ_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from deployment.inference import SteganalysiPredictor
from defense_context.risk_scorer import compute_risk_score, TransferMetadata, RISK_COLORS

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StegShield — Steganography Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.main-header h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: rgba(255,255,255,0.7);
    font-size: 1rem;
    margin: 0;
}

.verdict-card {
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    animation: pulse 2s infinite;
}

.verdict-clean {
    background: linear-gradient(135deg, #1a472a, #2d6a4f);
    color: #52b788;
    border: 2px solid #52b788;
}

.verdict-stego {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: #f87171;
    border: 2px solid #f87171;
}

.metric-card {
    background: #1e1e2e;
    border: 1px solid #2d2d3d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}

.metric-card h4 {
    color: #a0aec0;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 0.3rem 0;
}

.metric-card h2 {
    color: white;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}

.risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
}

.section-title {
    color: white;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2d2d3d;
}

.factor-item {
    background: #2a2a3e;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 1rem;
    margin: 0.4rem 0;
    color: #c4c4d4;
    font-size: 0.9rem;
}

.info-box {
    background: #1a1a2e;
    border: 1px solid #2d2d3d;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
    color: #9ca3af;
    font-size: 0.85rem;
    line-height: 1.6;
}

.branch-bar-container {
    margin: 0.5rem 0;
}

.branch-label {
    color: #a0aec0;
    font-size: 0.8rem;
    margin-bottom: 0.2rem;
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
    50% { box-shadow: 0 4px 40px rgba(0,0,0,0.3); }
}

/* Dark mode streamlit overrides */
.stApp { background-color: #0d0d1a; }
div[data-testid="stSidebar"] { background-color: #12121f; }
</style>
""", unsafe_allow_html=True)


# ── Model Loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def load_predictor():
    """Load model once and cache it. Raises a visible error if checkpoint is missing."""
    checkpoint = os.path.join(PROJ_ROOT, "weights", "fusion_best.pt")

    if not os.path.exists(checkpoint):
        st.error(
            "⛔ **Model checkpoint not found.**\n\n"
            f"Expected: `{checkpoint}`\n\n"
            "Train the model first, then restart the demo:\n"
            "```bash\n"
            "python src/training/train.py --mode branch_a\n"
            "python src/training/train.py --mode branch_b\n"
            "python src/training/train.py --mode branch_c\n"
            "python src/training/train.py --mode fusion\n"
            "```\n"
            "Or download pretrained weights from the Google Colab notebook: "
            "`notebooks/05_fusion_training.ipynb`"
        )
        st.stop()   # halts execution — no random-weight predictions shown

    return SteganalysiPredictor(checkpoint=checkpoint)


# ── Helper Functions ──────────────────────────────────────────────────────────

def render_confidence_bar(label: str, value: float, color: str):
    """Render a horizontal confidence bar with HTML."""
    bar_width = int(value * 100)
    st.markdown(f"""
    <div class="branch-bar-container">
        <div class="branch-label">{label}: {value*100:.1f}%</div>
        <div style="background:#2d2d3d; border-radius:999px; height:8px; width:100%;">
            <div style="background:{color}; width:{bar_width}%; height:100%;
                        border-radius:999px; transition: width 0.5s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def risk_level_emoji(level: str) -> str:
    return {"CLEAN": "✅", "LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "🚨"}.get(level, "⚪")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ StegShield")
    st.markdown("Multi-Branch CNN Steganography Detector")
    st.divider()

    st.markdown("### How It Works")
    st.markdown("""
**Three inspection branches run in parallel:**

🔬 **Branch A — Pixel CNN**  
Detects LSB steganography via SRM filters

📡 **Branch B — Frequency CNN**  
Detects JPEG-domain stego via DCT analysis

📊 **Branch C — Statistics MLP**  
Detects classical statistical anomalies

🔀 **Fusion Layer**  
Combines all three → final verdict
    """)
    st.divider()

    st.markdown("### Defense Context")
    st.markdown("*Optional: provide transfer metadata for risk scoring*")

    # Metadata inputs
    user_role = st.selectbox(
        "User Role",
        ["admin", "normal", "intern", "guest", "unknown"],
        index=1
    )

    transfer_time = st.time_input("Transfer Time", value=None)

    dest_ip = st.text_input(
        "Destination IP",
        placeholder="e.g. 103.45.67.89 (leave blank if unknown)"
    )

    file_size_mb = st.number_input(
        "Actual File Size (MB)",
        min_value=0.0, value=0.0, step=0.1
    )

    expected_size_mb = st.number_input(
        "Expected File Size (MB)",
        min_value=0.0, value=0.0, step=0.1
    )

    is_repeat = st.checkbox("User previously flagged?", value=False)

    st.divider()
    st.caption("📝 EDI Project | 2nd Year CSE")
    st.caption("Multi-Branch CNN + Defense-Context Scoring")


# ── Main Layout ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🛡️ StegShield</h1>
    <p>Steganography Detection Using Multi-Branch CNN &nbsp;|&nbsp; Defense Intelligence System</p>
</div>
""", unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown('<div class="section-title">📁 Upload Image for Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop an image here or click to browse",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Show image stats
        img_array = np.array(image)
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Resolution", f"{image.size[0]}×{image.size[1]}")
        col_s2.metric("File Size", f"{uploaded_file.size/1024:.1f} KB")
        col_s3.metric("Mode", "Grayscale")

        st.markdown('<div class="info-box">ℹ️ Image is analyzed through three parallel CNN branches. '
                    'The model was trained on the BOSS dataset with 10,000 images (50% stego / 50% clean).</div>',
                    unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="border: 2px dashed #2d2d3d; border-radius: 16px; padding: 3rem;
                    text-align: center; color: #4a4a6a; margin-top: 1rem;">
            <div style="font-size: 3rem;">🖼️</div>
            <div style="font-size: 1.1rem; margin-top: 0.5rem;">Upload a PNG, JPG, or BMP image</div>
            <div style="font-size: 0.8rem; margin-top: 0.3rem; color: #3a3a5a;">
                Works with any image. Suspicious ones will be flagged.
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="section-title">🔍 Analysis Results</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("Running multi-branch analysis..."):
            try:
                predictor = load_predictor()
                result    = predictor.predict(image)

                # ── Verdict Card ──────────────────────────────────────────────
                is_stego = result["label"] == "STEGO"
                verdict_class = "verdict-stego" if is_stego else "verdict-clean"
                verdict_icon  = "⚠️ STEGO DETECTED" if is_stego else "✅ IMAGE CLEAN"

                st.markdown(f"""
                <div class="verdict-card {verdict_class}">
                    {verdict_icon}
                </div>
                """, unsafe_allow_html=True)

                # ── Confidence Metrics ────────────────────────────────────────
                c1, c2, c3 = st.columns(3)
                c1.metric("Stego Probability", f"{result['stego_prob']*100:.1f}%")
                c2.metric("Confidence",        f"{result['confidence']*100:.1f}%")
                c3.metric("Inference Time",    f"{result['inference_ms']} ms")

                # ── Confidence Bars ───────────────────────────────────────────
                st.markdown("")
                render_confidence_bar("Stego Probability",  result["stego_prob"],  "#E74C3C")
                render_confidence_bar("Clean Probability",  result["clean_prob"],  "#27AE60")

                # ── Defense-Context Risk Scoring ──────────────────────────────
                st.markdown('<div class="section-title">🔐 Defense-Context Risk Assessment</div>',
                            unsafe_allow_html=True)

                metadata = TransferMetadata(
                    user_role        = user_role,
                    timestamp        = datetime.combine(datetime.today().date(), transfer_time)
                                       if transfer_time else None,
                    file_size_bytes  = int(file_size_mb * 1024 * 1024) if file_size_mb > 0 else None,
                    expected_size_bytes = int(expected_size_mb * 1024 * 1024) if expected_size_mb > 0 else None,
                    destination_ip   = dest_ip if dest_ip.strip() else None,
                    is_repeat_offender = is_repeat
                )

                risk = compute_risk_score(result["stego_prob"], metadata)
                risk_emoji = risk_level_emoji(risk.risk_level)

                # Risk level display
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.markdown(f"""
                    <div style="text-align:center; padding: 1.5rem; background: #1e1e2e;
                                border-radius: 12px; border: 1px solid #2d2d3d;">
                        <div style="font-size: 2.5rem;">{risk_emoji}</div>
                        <div style="color: {risk.risk_color}; font-weight: 700;
                                    font-size: 1.3rem; margin-top: 0.3rem;">
                            {risk.risk_level}
                        </div>
                        <div style="color: #6b7280; font-size: 0.8rem; margin-top: 0.2rem;">
                            Score: {risk.risk_score}/100
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_r2:
                    # Risk score bar
                    st.markdown("**Risk Score**")
                    render_confidence_bar("", risk.risk_score / 100, risk.risk_color)
                    st.caption(risk.alert_message)

                # Contributing factors
                if risk.contributing_factors:
                    st.markdown("**Contributing Risk Factors:**")
                    for factor in risk.contributing_factors:
                        st.markdown(f'<div class="factor-item">⚡ {factor}</div>',
                                    unsafe_allow_html=True)
                else:
                    st.success("No risk factors detected.")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("💡 Tip: Make sure the model is trained and saved at `weights/fusion_best.pt`")
                st.code(str(e))

    else:
        st.markdown("""
        <div style="border: 1px solid #2d2d3d; border-radius: 16px; padding: 4rem 2rem;
                    text-align: center; color: #4a4a6a; height: 100%;">
            <div style="font-size: 3rem;">🛡️</div>
            <div style="font-size: 1.1rem; margin-top: 0.8rem;">
                Upload an image to begin analysis
            </div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem; color: #3a3a5a; max-width: 300px; margin: 0.5rem auto 0;">
                The three-branch CNN will inspect pixel residuals,
                DCT frequency patterns, and statistical features simultaneously.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; color: #4a4a6a; font-size: 0.8rem; padding: 1rem 0;">
    <strong>StegShield</strong> &nbsp;·&nbsp;
    Multi-Branch CNN Steganography Detector &nbsp;·&nbsp;
    Branch A (Pixel) + Branch B (DCT) + Branch C (Statistics) &nbsp;·&nbsp;
    INT8 CPU-Deployable &nbsp;·&nbsp;
    2nd Year CSE EDI Project
</div>
""", unsafe_allow_html=True)
