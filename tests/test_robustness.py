from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from chaos_image_encryption.attacks import block_erasure, jpeg_recompress, salt_pepper_noise
from chaos_image_encryption.evaluation.robustness import pixel_error_rate, run_key_sensitivity, run_robustness


def _sample_rgb(h=32, w=32):
    y, x = np.mgrid[:h, :w]
    return np.stack([
        (x * 7 + y * 3) % 256,
        (x * 5 + y * 11) % 256,
        (x * 13 + y * 2) % 256,
    ], axis=-1).astype(np.uint8)


def _write_image(path: Path, arr: np.ndarray):
    Image.fromarray(arr, mode="RGB").save(path)


def test_salt_pepper_noise_changes_requested_bytes():
    arr = np.full((20, 20, 3), 127, dtype=np.uint8)
    out = salt_pepper_noise(arr, 0.10, seed=1)
    assert out.shape == arr.shape
    assert out.dtype == np.uint8
    assert np.count_nonzero(out != arr) == round(arr.size * 0.10)


def test_block_erasure_preserves_shape_and_changes_region():
    arr = np.full((20, 30, 3), 255, dtype=np.uint8)
    out = block_erasure(arr, 0.25, seed=2)
    assert out.shape == arr.shape
    assert np.count_nonzero(out != arr) > 0
    assert np.any(out == 0)


def test_jpeg_recompress_preserves_shape_and_dtype():
    arr = _sample_rgb(48, 64)
    out = jpeg_recompress(arr, 60)
    assert out.shape == arr.shape
    assert out.dtype == np.uint8
    assert np.any(out != arr)


def test_pixel_error_rate():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = a.copy()
    b[0, :10] = 1
    assert pixel_error_rate(a, b) == 10.0


def test_run_robustness_smoke(tmp_path):
    image_path = tmp_path / "sample.png"
    _write_image(image_path, _sample_rgb())
    out = tmp_path / "robustness"
    payload = run_robustness(
        image_path,
        out,
        crypto="ecc",
        noise_levels=(0.01,),
        block_levels=(0.05,),
        jpeg_qualities=(70,),
        seed=7,
        save_images=False,
    )
    assert len(payload["rows"]) == 4
    assert payload["rows"][0]["attack"] == "baseline"
    assert payload["rows"][0]["exact_match"] is True
    assert (out / "robustness.csv").exists()
    assert (out / "summary.json").exists()
    assert (out / "plots" / "recovered_ssim.png").exists()


def test_run_key_sensitivity_smoke(tmp_path):
    image_path = tmp_path / "sample.png"
    _write_image(image_path, _sample_rgb())
    out = tmp_path / "key_sensitivity"
    payload = run_key_sensitivity(image_path, out, trials=3, seed=9, save_images=False)
    assert len(payload["rows"]) == 3
    assert payload["summary"]["wrong_key_exact_match_count"] == 0
    assert payload["summary"]["cipher_npcr_percent_mean"] > 95.0
    assert 20.0 < payload["summary"]["cipher_uaci_percent_mean"] < 45.0
    assert (out / "trials.csv").exists()
    assert (out / "summary.json").exists()
    assert (out / "plots" / "npcr_by_trial.png").exists()
