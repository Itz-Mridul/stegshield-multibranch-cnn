"""
run_all_tests.py — Master self-test script for all modules.
Run this after training to verify everything works end-to-end.

Usage: python3 run_all_tests.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
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

print("\n" + "="*55)
print("  StegShield — Full Module Self-Test Suite")
print("="*55 + "\n")

# ── Test 1: LSB Embedder ──────────────────────────────────────────────────────
print("[ 1/6 ] LSB Embedder")
def test_lsb():
    from data_pipeline.lsb_embedder import embed_lsb, extract_lsb
    dummy = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    payload = b"Test payload for steganography verification 12345"
    stego = embed_lsb(dummy, payload)
    recovered = extract_lsb(stego, len(payload))
    assert recovered == payload, f"Payload mismatch"
    max_diff = np.max(np.abs(dummy.astype(int) - stego.astype(int)))
    assert max_diff <= 1, f"Max pixel diff {max_diff} > 1"
run_test("embed_lsb + extract_lsb", test_lsb)

# ── Test 2: SRM Filters ───────────────────────────────────────────────────────
print("\n[ 2/6 ] SRM Filter Layer")
def test_srm():
    from model.srm_filters import get_srm_filters, SRMFilterLayer
    filters = get_srm_filters()
    assert filters.shape == torch.Size([30, 1, 5, 5]), f"Shape: {filters.shape}"
    layer = SRMFilterLayer()
    x = torch.randn(2, 1, 256, 256)
    out = layer(x)
    assert out.shape == torch.Size([2, 30, 256, 256]), f"Output: {out.shape}"
    frozen = all(not p.requires_grad for p in layer.parameters())
    assert frozen, "SRM weights should be frozen (requires_grad=False)"
run_test("SRM filters shape + frozen weights", test_srm)

# ── Test 3: Branch A ──────────────────────────────────────────────────────────
print("\n[ 3/6 ] Branch A — Pixel CNN")
def test_branch_a():
    from branches.branch_a_pixel import BranchAClassifier, PixelCNN
    batch = torch.randn(2, 1, 256, 256)

    # Test standalone classifier
    model = BranchAClassifier(feature_dim=256, num_classes=2)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        features = model.get_features(batch)
    assert logits.shape  == torch.Size([2, 2]),   f"Logits: {logits.shape}"
    assert features.shape == torch.Size([2, 256]), f"Features: {features.shape}"

    # Check SRM weights are frozen inside branch
    srm_params = list(model.backbone.srm.parameters())
    assert all(not p.requires_grad for p in srm_params), "SRM not frozen in Branch A!"
run_test("BranchAClassifier forward + feature extraction", test_branch_a)

# ── Test 4: Branch B ──────────────────────────────────────────────────────────
print("\n[ 4/6 ] Branch B — DCT Frequency CNN")
def test_branch_b():
    from branches.branch_b_dct import BranchBClassifier, image_to_dct_tensor
    batch = torch.randn(2, 1, 256, 256)

    # Test DCT conversion
    dct = image_to_dct_tensor(batch)
    assert dct.shape == torch.Size([2, 1, 256, 256]), f"DCT shape: {dct.shape}"
    assert not torch.isnan(dct).any(), "NaN in DCT output!"

    # Test standalone classifier
    model = BranchBClassifier(feature_dim=256, num_classes=2)
    model.eval()
    with torch.no_grad():
        logits   = model(batch)
        features = model.get_features(batch)
    assert logits.shape   == torch.Size([2, 2]),   f"Logits: {logits.shape}"
    assert features.shape == torch.Size([2, 256]), f"Features: {features.shape}"
run_test("BranchBClassifier + DCT conversion", test_branch_b)

# ── Test 5: Branch C ──────────────────────────────────────────────────────────
print("\n[ 5/6 ] Branch C — Statistical MLP")
def test_branch_c():
    from branches.branch_c_stats import (
        BranchCClassifier, extract_statistical_features, FEATURE_DIM
    )
    # Test feature extraction
    dummy_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    feat = extract_statistical_features(dummy_img)
    assert len(feat) == FEATURE_DIM, f"Feature dim: {len(feat)} != {FEATURE_DIM}"
    assert not np.isnan(feat).any(), "NaN in statistical features!"

    # Test model
    batch = torch.randn(2, 1, 256, 256)
    model = BranchCClassifier(feature_dim=32, num_classes=2)
    model.eval()
    with torch.no_grad():
        logits   = model(batch)
        features = model.get_features(batch)
    assert logits.shape   == torch.Size([2, 2]),  f"Logits: {logits.shape}"
    assert features.shape == torch.Size([2, 32]), f"Features: {features.shape}"
run_test("BranchCClassifier + feature extraction", test_branch_c)

# ── Test 6: Full Fusion Model ─────────────────────────────────────────────────
print("\n[ 6/6 ] Full Fusion Model (MultiBranchSteganalyzer)")
def test_fusion():
    from model.fusion_model import MultiBranchSteganalyzer
    batch = torch.randn(2, 1, 256, 256)

    model = MultiBranchSteganalyzer(
        feature_dim_a=256, feature_dim_b=256, feature_dim_c=32
    )
    model.eval()
    with torch.no_grad():
        # Test forward pass with branch features
        logits, fa, fb, fc = model(batch, return_branch_features=True)
        assert logits.shape == torch.Size([2, 2]),   f"Logits: {logits.shape}"
        assert fa.shape     == torch.Size([2, 256]), f"Feat A: {fa.shape}"
        assert fb.shape     == torch.Size([2, 256]), f"Feat B: {fb.shape}"
        assert fc.shape     == torch.Size([2, 32]),  f"Feat C: {fc.shape}"

        # Test predict()
        result = model.predict(batch)
        assert 'label'      in result, "Missing 'label'"
        assert 'confidence' in result, "Missing 'confidence'"
        assert 'stego_prob' in result, "Missing 'stego_prob'"
        assert all(l in ['CLEAN','STEGO'] for l in result['label'])
        assert all(0 <= p <= 1 for p in result['confidence'])

    # Parameter count sanity check
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    assert trainable < total, "All params trainable — SRM freeze failed!"

run_test("MultiBranchSteganalyzer forward + predict()", test_fusion)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
passed = sum(results.values())
total  = len(results)
print(f"  Results: {passed}/{total} tests passed")
print("="*55)

for name, ok in results.items():
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}")

print()
if passed == total:
    print("  🎉 ALL TESTS PASSED — project is ready!")
    print("  Next step: bash scripts/setup_boss_dataset.sh")
else:
    print(f"  ⚠️  {total - passed} test(s) failed — check errors above.")

print()
