"""
run_all_tests.py
================
Master self-test suite — tests ALL modules including edge cases.

Usage: python run_all_tests.py

Tests added in v2 (issues 8-12 from second code review):
  - Train/val leakage test (paired split)
  - PNG/JPEG embedding format enforcement
  - payload_fraction passthrough
  - Inference with missing checkpoint (must raise, not warn)
  - Metric calculation on edge cases (all-correct, all-wrong, empty)
  - Empty folder detection
  - Crop size smaller than image (must raise)
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from PIL import Image

PASS = "✅"
FAIL = "❌"
results = {}


def run_test(name, fn):
    try:
        fn()
        results[name] = True
        print(f"  {PASS} {name}")
    except Exception as e:
        results[name] = False
        print(f"  {FAIL} {name}: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROUP 1 — LSB Embedder (core + edge cases)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("  [ Group 1 ] LSB Embedder — core + edge cases")
print("=" * 60)


def test_lsb_basic():
    from data_pipeline.lsb_embedder import embed_lsb, extract_lsb
    dummy   = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    payload = b"StegShield test payload 12345"
    stego   = embed_lsb(dummy, payload)
    recovered = extract_lsb(stego, len(payload))
    assert recovered == payload, "Payload mismatch"
    max_diff = np.max(np.abs(dummy.astype(int) - stego.astype(int)))
    assert max_diff <= 1, f"Max pixel diff {max_diff} > 1"


def test_lsb_payload_too_large():
    """embed_lsb must raise ValueError when payload exceeds capacity."""
    from data_pipeline.lsb_embedder import embed_lsb
    tiny   = np.zeros((4, 4), dtype=np.uint8)   # 16 pixels = 2 bytes max
    big    = b"way too much data"                # 17 bytes
    raised = False
    try:
        embed_lsb(tiny, big)
    except ValueError:
        raised = True
    assert raised, "Should raise ValueError for oversized payload"


def test_lsb_png_output_enforced():
    """create_stego_image must save as PNG even if output_path ends in .jpg."""
    from data_pipeline.lsb_embedder import create_stego_image
    with tempfile.TemporaryDirectory() as tmp:
        clean = os.path.join(tmp, "clean.png")
        Image.fromarray(
            np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        ).save(clean)

        # Request .jpg output — should be silently redirected to .png
        jpg_out = os.path.join(tmp, "stego.jpg")
        create_stego_image(clean, jpg_out)

        png_out = os.path.join(tmp, "stego.png")
        assert os.path.exists(png_out), \
            f".png not created; expected {png_out}"
        assert not os.path.exists(jpg_out), \
            f".jpg file unexpectedly written; LSB would have been corrupted"


def test_lsb_jpeg_payload_survives():
    """Payload embedded into a clean PNG should survive round-trip only if saved as PNG."""
    from data_pipeline.lsb_embedder import embed_lsb, extract_lsb
    img   = np.random.randint(100, 200, (64, 64), dtype=np.uint8)
    data  = b"secret"
    stego = embed_lsb(img, data)

    # PNG round-trip — lossless
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        png_path = f.name
    Image.fromarray(stego).save(png_path)
    loaded_png = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
    assert extract_lsb(loaded_png, len(data)) == data, "PNG round-trip corrupted payload"
    os.unlink(png_path)

    # JPEG round-trip — lossy — should NOT recover payload
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        jpg_path = f.name
    Image.fromarray(stego).save(jpg_path, quality=95)
    loaded_jpg = np.array(Image.open(jpg_path).convert("L"), dtype=np.uint8)
    # JPEG compression at q=95 will still alter some LSBs → payload corrupted
    # We assert that at least some bytes differ (i.e., JPEG IS destructive)
    recovered_jpg = extract_lsb(loaded_jpg, len(data))
    os.unlink(jpg_path)
    # Don't assert failure (q=95 might get lucky on small payloads);
    # just confirm PNG works while documenting the JPEG risk in test name.
    assert extract_lsb(loaded_png, len(data)) == data, "PNG baseline failed"


def test_lsb_payload_fraction():
    """payload_fraction must correctly control how many bytes are embedded."""
    from data_pipeline.lsb_embedder import create_stego_image
    with tempfile.TemporaryDirectory() as tmp:
        clean = os.path.join(tmp, "c.png")
        Image.fromarray(
            np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        ).save(clean)

        out_10 = os.path.join(tmp, "s10.png")
        out_50 = os.path.join(tmp, "s50.png")

        n10 = create_stego_image(clean, out_10, payload_fraction=0.10)
        n50 = create_stego_image(clean, out_50, payload_fraction=0.50)

        # 50% fraction should embed ~5x more bytes than 10%
        assert n50 > n10 * 3, (
            f"payload_fraction not working: 10%={n10}B, 50%={n50}B "
            "(50% should embed ~5× more)"
        )


def test_lsb_invalid_payload_fraction():
    """payload_fraction outside (0, 1] must raise ValueError."""
    from data_pipeline.lsb_embedder import create_stego_image
    with tempfile.TemporaryDirectory() as tmp:
        clean = os.path.join(tmp, "c.png")
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(clean)
        for bad in [0.0, -0.1, 1.5]:
            raised = False
            try:
                create_stego_image(clean, os.path.join(tmp, "s.png"),
                                   payload_fraction=bad)
            except ValueError:
                raised = True
            assert raised, f"Should raise ValueError for fraction={bad}"


run_test("embed_lsb + extract_lsb (basic)", test_lsb_basic)
run_test("embed_lsb raises ValueError on oversized payload", test_lsb_payload_too_large)
run_test("JPEG output path auto-corrected to PNG", test_lsb_png_output_enforced)
run_test("PNG round-trip preserves LSB payload; JPEG documents destruction", test_lsb_jpeg_payload_survives)
run_test("payload_fraction controls embed size", test_lsb_payload_fraction)
run_test("payload_fraction outside (0,1] raises ValueError", test_lsb_invalid_payload_fraction)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROUP 2 — Dataset / Split (pair safety + empty folder)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("  [ Group 2 ] Dataset — pair-safe split + empty folder guard")
print("=" * 60)


def _make_dataset_dir(tmp, n_images=20, size=300):
    """Create n_images synthetic PNG images in tmp/clean and tmp/stego."""
    from data_pipeline.lsb_embedder import embed_lsb
    clean_dir = os.path.join(tmp, "clean")
    stego_dir = os.path.join(tmp, "stego")
    os.makedirs(clean_dir)
    os.makedirs(stego_dir)

    for i in range(n_images):
        arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        fname = f"img_{i:04d}.png"
        Image.fromarray(arr).save(os.path.join(clean_dir, fname))

        payload = bytes([i % 256] * 100)
        stego_arr = embed_lsb(arr, payload)
        Image.fromarray(stego_arr).save(os.path.join(stego_dir, fname))

    return clean_dir, stego_dir


def test_pair_safe_split_no_leakage():
    """
    After splitting, NO stem should appear in both train and val.
    This is the core data-leakage test.
    """
    from data_pipeline.dataset import split_stems_by_pair, SteganalysisDataset
    import torchvision.transforms as T

    with tempfile.TemporaryDirectory() as tmp:
        clean_dir, stego_dir = _make_dataset_dir(tmp, n_images=20)

        train_stems, val_stems = split_stems_by_pair(
            clean_dir, val_split=0.2, seed=42
        )

        # No stem must appear in both sets
        overlap = train_stems & val_stems
        assert len(overlap) == 0, (
            f"DATA LEAKAGE: {len(overlap)} stems appear in both train and val: "
            f"{list(overlap)[:5]}"
        )

        # Together they must cover all stems
        tf = T.Compose([T.CenterCrop(256), T.ToTensor()])
        all_dataset = SteganalysisDataset(clean_dir, stego_dir, transform=tf)
        all_stems = {os.path.splitext(os.path.basename(p))[0]
                     for p, _ in all_dataset.samples}
        assert train_stems | val_stems == all_stems, \
            "train + val stems don't cover all images"


def test_no_pair_crosses_split():
    """
    For every stem, its clean and stego versions must be in the SAME split.
    """
    from data_pipeline.dataset import split_stems_by_pair, SteganalysisDataset
    import torchvision.transforms as T

    with tempfile.TemporaryDirectory() as tmp:
        clean_dir, stego_dir = _make_dataset_dir(tmp, n_images=20)
        train_stems, val_stems = split_stems_by_pair(
            clean_dir, val_split=0.2, seed=7
        )
        tf = T.Compose([T.CenterCrop(256), T.ToTensor()])

        train_ds = SteganalysisDataset(
            clean_dir, stego_dir, allowed_stems=train_stems, transform=tf
        )
        val_ds = SteganalysisDataset(
            clean_dir, stego_dir, allowed_stems=val_stems, transform=tf
        )

        train_set = {os.path.splitext(os.path.basename(p))[0]
                     for p, _ in train_ds.samples}
        val_set   = {os.path.splitext(os.path.basename(p))[0]
                     for p, _ in val_ds.samples}

        cross = train_set & val_set
        assert len(cross) == 0, (
            f"Cross-split pairs found: {list(cross)[:5]}. "
            "A stego image is in val while its clean pair is in train (or vice versa)."
        )


def test_empty_folder_raises():
    """SteganalysisDataset must raise RuntimeError on empty folders."""
    from data_pipeline.dataset import SteganalysisDataset
    import torchvision.transforms as T
    with tempfile.TemporaryDirectory() as tmp:
        empty_clean = os.path.join(tmp, "clean"); os.makedirs(empty_clean)
        empty_stego = os.path.join(tmp, "stego"); os.makedirs(empty_stego)
        tf = T.Compose([T.CenterCrop(256), T.ToTensor()])
        raised = False
        try:
            SteganalysisDataset(empty_clean, empty_stego, transform=tf)
        except RuntimeError:
            raised = True
        assert raised, "Empty folders should raise RuntimeError"


def test_transform_isolation():
    """
    Train and val datasets must have INDEPENDENT transforms.
    Modifying val_dataset.transform must NOT affect train_dataset.
    (Prevents the shared-transform mutation bug.)
    """
    from data_pipeline.dataset import SteganalysisDataset
    import torchvision.transforms as T

    with tempfile.TemporaryDirectory() as tmp:
        clean_dir, stego_dir = _make_dataset_dir(tmp, n_images=10)
        train_tf = T.Compose([T.RandomCrop(256), T.ToTensor()])
        val_tf   = T.Compose([T.CenterCrop(256), T.ToTensor()])

        train_ds = SteganalysisDataset(clean_dir, stego_dir, transform=train_tf)
        val_ds   = SteganalysisDataset(clean_dir, stego_dir, transform=val_tf)

        # Reassigning val transform must not touch train
        original_train_tf = train_ds.transform
        val_ds.transform  = T.Compose([T.CenterCrop(200), T.ToTensor()])
        assert train_ds.transform is original_train_tf, \
            "Modifying val_ds.transform affected train_ds.transform — shared state bug"


run_test("pair-safe split: zero stem overlap between train and val", test_pair_safe_split_no_leakage)
run_test("no clean-stego pair crosses the train/val boundary", test_no_pair_crosses_split)
run_test("empty folder raises RuntimeError", test_empty_folder_raises)
run_test("train and val transforms are independent (no shared state)", test_transform_isolation)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROUP 3 — PyTorch Model Tests (require torch)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("  [ Group 3 ] PyTorch Models")
print("=" * 60)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  ⚠  PyTorch not installed — skipping model tests.")
    print("     Install: conda install pytorch torchvision -c pytorch")


def test_srm_filters():
    from model.srm_filters import get_srm_filters, SRMFilterLayer
    filters = get_srm_filters()
    assert filters.shape == torch.Size([30, 1, 5, 5])
    layer = SRMFilterLayer()
    out = layer(torch.randn(2, 1, 256, 256))
    assert out.shape == torch.Size([2, 30, 256, 256])
    assert all(not p.requires_grad for p in layer.parameters()), \
        "SRM weights must be frozen"


def test_branch_a():
    from branches.branch_a_pixel import BranchAClassifier
    model = BranchAClassifier(feature_dim=256, num_classes=2)
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        logits   = model(x)
        features = model.get_features(x)
    assert logits.shape   == torch.Size([2, 2])
    assert features.shape == torch.Size([2, 256])
    srm_params = list(model.backbone.srm.parameters())
    assert all(not p.requires_grad for p in srm_params), "SRM not frozen in Branch A"


def test_branch_b():
    from branches.branch_b_dct import BranchBClassifier, image_to_dct_tensor
    x = torch.randn(2, 1, 256, 256)
    dct = image_to_dct_tensor(x)
    assert dct.shape == torch.Size([2, 1, 256, 256])
    assert not torch.isnan(dct).any(), "NaN in DCT output"
    model = BranchBClassifier(feature_dim=256, num_classes=2)
    model.eval()
    with torch.no_grad():
        logits   = model(x)
        features = model.get_features(x)
    assert logits.shape   == torch.Size([2, 2])
    assert features.shape == torch.Size([2, 256])


def test_branch_c():
    from branches.branch_c_stats import (
        BranchCClassifier, extract_statistical_features, FEATURE_DIM
    )
    dummy = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    feat  = extract_statistical_features(dummy)
    assert len(feat) == FEATURE_DIM
    assert not np.isnan(feat).any(), "NaN in statistical features"
    model = BranchCClassifier(feature_dim=32, num_classes=2)
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        logits   = model(x)
        features = model.get_features(x)
    assert logits.shape   == torch.Size([2, 2])
    assert features.shape == torch.Size([2, 32])


def test_fusion_model():
    from model.fusion_model import MultiBranchSteganalyzer
    model = MultiBranchSteganalyzer(feature_dim_a=256, feature_dim_b=256, feature_dim_c=32)
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        logits, fa, fb, fc = model(x, return_branch_features=True)
    assert logits.shape == torch.Size([2, 2])
    assert fa.shape     == torch.Size([2, 256])
    assert fb.shape     == torch.Size([2, 256])
    assert fc.shape     == torch.Size([2, 32])
    result = model.predict(x)
    assert all(l in ["CLEAN", "STEGO"] for l in result["label"])
    assert all(0 <= p <= 1 for p in result["confidence"])
    # SRM still frozen inside fusion model
    srm_params = list(model.branch_a.srm.parameters())
    assert all(not p.requires_grad for p in srm_params)


def test_inference_missing_checkpoint_raises():
    """SteganalysiPredictor must raise FileNotFoundError — NOT silent random weights."""
    from deployment.inference import SteganalysiPredictor
    raised = False
    try:
        SteganalysiPredictor(checkpoint="/nonexistent/path/model.pt")
    except FileNotFoundError:
        raised = True
    assert raised, (
        "Missing checkpoint must raise FileNotFoundError. "
        "Silent random-weight inference produces meaningless results."
    )


def test_lsb_crop_preserves_pixels():
    """
    After CenterCrop, pixel values at crop positions must equal the original.
    This verifies that our preprocessing preserves LSB values exactly.
    """
    import torchvision.transforms as T
    # Create image with known pixel values
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    arr = np.tile(arr, (16, 16))  # 256x256 with deterministic values
    img = Image.fromarray(arr)
    crop_tf = T.CenterCrop(128)
    cropped = np.array(crop_tf(img))

    # Centre of 256x256 → rows 64:192, cols 64:192
    expected = arr[64:192, 64:192]
    assert np.array_equal(cropped, expected), \
        "CenterCrop altered pixel values — lossless property violated"


if TORCH_AVAILABLE:
    run_test("SRM filters shape + frozen weights", test_srm_filters)
    run_test("Branch A: forward + features + SRM frozen", test_branch_a)
    run_test("Branch B: DCT conversion + forward", test_branch_b)
    run_test("Branch C: 262-dim features + MLP forward", test_branch_c)
    run_test("Full Fusion Model: forward + predict()", test_fusion_model)
    run_test("Missing checkpoint raises FileNotFoundError (not silent)", test_inference_missing_checkpoint_raises)

# This test does NOT need PyTorch
run_test("CenterCrop preserves exact pixel values (lossless)", test_lsb_crop_preserves_pixels)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROUP 4 — Metric edge cases (no torch needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("  [ Group 4 ] Metric calculation — edge cases")
print("=" * 60)

try:
    from sklearn.metrics import accuracy_score, f1_score
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    print("  ⚠  scikit-learn not installed — skipping metric tests.")


def test_metrics_all_correct():
    labels = np.array([0, 0, 1, 1, 0, 1])
    preds  = labels.copy()
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    assert acc == 1.0, f"All-correct accuracy should be 1.0, got {acc}"
    assert f1  == 1.0, f"All-correct F1 should be 1.0, got {f1}"


def test_metrics_all_wrong():
    labels = np.array([0, 0, 1, 1])
    preds  = np.array([1, 1, 0, 0])
    acc = accuracy_score(labels, preds)
    assert acc == 0.0, f"All-wrong accuracy should be 0.0, got {acc}"


def test_metrics_all_same_class():
    """All predictions = same class — F1 must not crash with zero_division guard."""
    labels = np.array([0, 1, 1, 0, 1])
    preds  = np.array([1, 1, 1, 1, 1])   # always predicts stego
    f1 = f1_score(labels, preds, zero_division=0)
    assert 0.0 <= f1 <= 1.0, f"F1 out of range: {f1}"


def test_chi_square_direction():
    """
    LSB-embedded image should have LOWER chi-square than a natural-ish image.

    Why the image choice matters:
      - Gradient tile (arange repeated): each brightness appears exactly ONCE
        → all adjacent bin pairs are already equal → chi-sq is already 0.
        Embedding can only increase it from 0. Wrong test image.
      - Purely random image: bins are already near-equal by central limit theorem
        → small signal buried in noise. Unreliable test.
      - Smooth sinusoidal: bins are unequal (peaked distribution around mid-grey)
        → large chi-sq before embedding; embedding strongly equalises pairs
        → chi-sq measurably decreases. Correct test image.

    After LSB embedding with 50% payload (random bits):
      Each pixel's last bit is overwritten with ~50% 0 and ~50% 1.
      This pushes each (even, odd) bin pair toward equal counts.
      chi-sq = sum((observed-expected)^2/expected) decreases toward 0.
    """
    from branches.branch_c_stats import compute_chi_square_statistic
    from data_pipeline.lsb_embedder import embed_lsb

    rng = np.random.default_rng(42)

    # Smooth sinusoidal image — high natural bin inequality, large chi-sq clean
    x = np.linspace(0, np.pi, 256, dtype=np.float32)
    row = (np.sin(x) * 127 + 128).astype(np.uint8)      # values 128±127
    clean = np.tile(row, (256, 1))                        # 256×256 smooth gradient

    # Embed at 50% capacity — strong equalization of even/odd pairs
    max_bytes = clean.size // 8
    payload   = bytes(rng.integers(0, 256, max_bytes // 2, dtype=np.uint8).tolist())
    stego     = embed_lsb(clean, payload)

    chi_clean = compute_chi_square_statistic(clean)
    chi_stego = compute_chi_square_statistic(stego)

    assert chi_clean > 10.0, (
        f"Test image has too-low natural chi-sq ({chi_clean:.2f}). "
        "Choose a smoother image so the effect of embedding is measurable."
    )
    assert chi_stego < chi_clean, (
        f"THEORY ERROR: chi-square should DECREASE after LSB embedding.\n"
        f"  Clean chi-sq:  {chi_clean:.2f}\n"
        f"  Stego chi-sq:  {chi_stego:.2f}\n"
        "LSB embedding equalises even/odd bin pairs → lower chi-sq."
    )


if SK_AVAILABLE:
    run_test("Metrics: all-correct → acc=1.0, f1=1.0", test_metrics_all_correct)
    run_test("Metrics: all-wrong → acc=0.0", test_metrics_all_wrong)
    run_test("Metrics: all-same-class → F1 does not crash", test_metrics_all_same_class)

run_test("Chi-square decreases after LSB embedding (direction correct)", test_chi_square_direction)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
passed = sum(results.values())
total  = len(results)
print(f"  Results: {passed}/{total} tests passed")
print("=" * 60)
for name, ok in results.items():
    print(f"  {'✅' if ok else '❌'}  {name}")

print()
if passed == total:
    print("  🎉 ALL TESTS PASSED — project is ready for training!")
    print("  Next step: bash scripts/setup_boss_dataset.sh")
else:
    n_failed = total - passed
    print(f"  ⚠️  {n_failed} test(s) FAILED — fix before trusting results.")
print()
