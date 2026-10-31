import numpy as np

from chaos_image_encryption.evaluation.metrics import (
    adjacent_correlations,
    entropy,
    mse,
    npcr,
    psnr,
    ssim,
    summarize_triplet,
    uaci,
)


def test_identical_metrics():
    r = np.random.default_rng(11)
    a = r.integers(0, 256, (32, 32), dtype=np.uint8)
    assert mse(a, a) == 0.0
    assert np.isinf(psnr(a, a))
    assert ssim(a, a) == 1.0


def test_entropy_range():
    r = np.random.default_rng(12)
    a = r.integers(0, 256, (128, 128), dtype=np.uint8)
    h = entropy(a)["overall"]
    assert 0.0 <= h <= 8.0


def test_correlations_keys():
    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    c = adjacent_correlations(a)
    assert set(c) == {"horizontal", "vertical", "diagonal"}


def test_npcr_uaci():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = a.copy()
    b[0, 0] = 255
    assert npcr(a, b) == 1.0
    assert uaci(a, b) == 1.0


def test_triplet_exact():
    r = np.random.default_rng(13)
    original = r.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    encrypted = np.roll(original, 1, axis=0)
    result = summarize_triplet(original, encrypted, original.copy())
    assert result["correctness"]["exact_match"] is True
    assert result["correctness"]["mse_original_decrypted"] == 0.0
