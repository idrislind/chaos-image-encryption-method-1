from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .metrics import summarize_triplet
from .plots import save_correlation_scatter, save_histogram


def _json_safe(value):
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_triplet_report(original, encrypted, decrypted, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = summarize_triplet(original, encrypted, decrypted)

    safe = _json_safe(metrics)
    (out / "metrics.json").write_text(json.dumps(safe, indent=2), encoding="utf-8")

    c = metrics["correctness"]
    eo = metrics["entropy"]["original"]
    ee = metrics["entropy"]["encrypted"]
    ed = metrics["entropy"]["decrypted"]
    co = metrics["correlation"]["original"]
    ce = metrics["correlation"]["encrypted"]
    cd = metrics["correlation"]["decrypted"]

    row = {
        "exact_match": c["exact_match"],
        "mse_original_decrypted": c["mse_original_decrypted"],
        "psnr_original_decrypted_db": "inf" if math.isinf(c["psnr_original_decrypted_db"]) else c["psnr_original_decrypted_db"],
        "ssim_original_decrypted": c["ssim_original_decrypted"],
        "entropy_original": eo["overall"],
        "entropy_encrypted": ee["overall"],
        "entropy_decrypted": ed["overall"],
        "corr_h_original": co["horizontal"],
        "corr_h_encrypted": ce["horizontal"],
        "corr_h_decrypted": cd["horizontal"],
        "corr_v_original": co["vertical"],
        "corr_v_encrypted": ce["vertical"],
        "corr_v_decrypted": cd["vertical"],
        "corr_d_original": co["diagonal"],
        "corr_d_encrypted": ce["diagonal"],
        "corr_d_decrypted": cd["diagonal"],
    }
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    for name, image in (("original", original), ("encrypted", encrypted), ("decrypted", decrypted)):
        save_histogram(image, out / "histograms" / f"{name}.png", f"Histogram - {name}")
        for direction in ("horizontal", "vertical", "diagonal"):
            save_correlation_scatter(
                image,
                out / "correlation" / f"{name}_{direction}.png",
                f"Adjacent-pixel correlation - {name} - {direction}",
                direction,
            )

    return metrics
