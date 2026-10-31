from pathlib import Path

import numpy as np
from PIL import Image

from chaos_image_encryption.evaluation.benchmark import _aggregate, discover_images, measure_call


def test_measure_call_returns_value_and_timing():
    m = measure_call(lambda x: x + 1, 4)
    assert m.value == 5
    assert m.elapsed_ms >= 0.0
    assert m.peak_rss_mib >= 0.0
    assert m.peak_rss_delta_mib >= 0.0


def test_discover_images_filters_supported_extensions(tmp_path):
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(tmp_path / "a.png")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(tmp_path / "b.pgm")
    (tmp_path / "notes.txt").write_text("x")
    found = discover_images(tmp_path)
    assert [p.name for p in found] == ["a.png", "b.pgm"]


def test_aggregate_mean_std_and_exact_rate():
    rows = [
        {"crypto": "rsa", "exact_match": True, "entropy_encrypted": 7.9, "npcr_percent": 99.5, "uaci_percent": 33.0,
         "corr_h_encrypted": 0.01, "corr_v_encrypted": 0.02, "corr_d_encrypted": 0.03,
         "session_create_ms": 1.0, "session_recover_ms": 2.0, "image_encrypt_ms": 10.0, "image_decrypt_ms": 11.0,
         "total_encrypt_ms": 11.0, "total_decrypt_ms": 13.0, "encrypt_throughput_mib_s": 2.0,
         "decrypt_throughput_mib_s": 1.8, "encrypt_peak_rss_delta_mib": 3.0, "decrypt_peak_rss_delta_mib": 2.5},
        {"crypto": "rsa", "exact_match": True, "entropy_encrypted": 8.0, "npcr_percent": 99.7, "uaci_percent": 34.0,
         "corr_h_encrypted": -0.01, "corr_v_encrypted": 0.0, "corr_d_encrypted": 0.01,
         "session_create_ms": 1.2, "session_recover_ms": 2.2, "image_encrypt_ms": 12.0, "image_decrypt_ms": 13.0,
         "total_encrypt_ms": 13.2, "total_decrypt_ms": 15.2, "encrypt_throughput_mib_s": 1.8,
         "decrypt_throughput_mib_s": 1.6, "encrypt_peak_rss_delta_mib": 4.0, "decrypt_peak_rss_delta_mib": 3.5},
    ]
    s = _aggregate(rows)[0]
    assert s["images"] == 2
    assert s["exact_match_rate_percent"] == 100.0
    assert abs(s["uaci_percent_mean"] - 33.5) < 1e-12
    assert s["image_encrypt_ms_min"] == 10.0
    assert s["image_encrypt_ms_max"] == 12.0
