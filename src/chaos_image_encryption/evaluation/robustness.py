from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..attacks import block_erasure, jpeg_recompress, salt_pepper_noise
from ..cipher.core import decrypt_array, encrypt_array
from ..crypto import ecc_backend, rsa_backend
from ..io.image_io import load_image, save_png
from .metrics import mse, npcr, psnr, ssim, uaci


def pixel_error_rate(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.uint8)
    bb = np.asarray(b, dtype=np.uint8)
    if aa.shape != bb.shape:
        raise ValueError(f"shape mismatch: {aa.shape} != {bb.shape}")
    return float(np.mean(aa != bb) * 100.0)


def _json_safe(value):
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if math.isnan(v):
            return "nan"
        return v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k, "")) for k in fields})


def _recovery_metrics(original: np.ndarray, recovered: np.ndarray) -> dict:
    return {
        "exact_match": bool(np.array_equal(original, recovered)),
        "mse_recovered": mse(original, recovered),
        "psnr_recovered_db": psnr(original, recovered),
        "ssim_recovered": ssim(original, recovered),
        "pixel_error_rate_percent": pixel_error_rate(original, recovered),
    }


def _create_backend_session(crypto: str, key_root: Path):
    if crypto == "rsa":
        private_key, public_key = rsa_backend.generate_keypair(key_root / "rsa")
        session_key, wrapped = rsa_backend.create_session(public_key)
        recovered_key = rsa_backend.recover_session(private_key, wrapped)
        overhead = len(wrapped)
    elif crypto == "ecc":
        private_key, public_key = ecc_backend.generate_keypair(key_root / "ecc")
        session_key, eph, salt = ecc_backend.create_session(public_key)
        recovered_key = ecc_backend.recover_session(private_key, eph, salt)
        overhead = len(eph) + len(salt)
    else:
        raise ValueError("crypto must be rsa or ecc")
    if recovered_key != session_key:
        raise RuntimeError(f"{crypto}: recovered session key differs")
    return recovered_key, overhead


def _scenario_row(
    original: np.ndarray,
    cipher: np.ndarray,
    attacked: np.ndarray,
    recovered: np.ndarray,
    *,
    crypto: str,
    attack: str,
    parameter: str,
    parameter_value: float | int,
) -> dict:
    row = {
        "crypto": crypto,
        "diffusion_version": "v3",
        "attack": attack,
        "parameter": parameter,
        "parameter_value": parameter_value,
        "cipher_npcr_percent": npcr(cipher, attacked),
        "cipher_uaci_percent": uaci(cipher, attacked),
    }
    row.update(_recovery_metrics(original, recovered))
    return row


def _save_robustness_plots(rows: list[dict], out_dir: Path) -> None:
    out = Path(out_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    attacked = [r for r in rows if r["attack"] != "baseline"]
    labels = [f"{r['attack']}:{r['parameter_value']}" for r in attacked]
    if not labels:
        return

    specs = [
        ("psnr_recovered_db", "Recovered PSNR (dB)", "recovered_psnr.png"),
        ("ssim_recovered", "Recovered SSIM", "recovered_ssim.png"),
        ("pixel_error_rate_percent", "Pixel error rate (%)", "pixel_error_rate.png"),
        ("cipher_npcr_percent", "Cipher bytes changed (%)", "cipher_change_percent.png"),
    ]
    for field, ylabel, filename in specs:
        vals = []
        for r in attacked:
            value = float(r[field])
            vals.append(np.nan if math.isinf(value) else value)
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4.5))
        ax.plot(range(len(labels)), vals, marker="o")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Attack scenario")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)


def run_robustness(
    image_path: Path,
    out_dir: Path,
    *,
    crypto: str = "rsa",
    noise_levels: tuple[float, ...] = (0.001, 0.005, 0.01),
    block_levels: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25),
    jpeg_qualities: tuple[int, ...] = (90, 70, 50),
    seed: int = 42,
    save_images: bool = True,
) -> dict:
    image, mode = load_image(image_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session_key, overhead = _create_backend_session(crypto, out / "keys")
    nonce = os.urandom(16)
    cipher = encrypt_array(image, session_key, nonce)
    baseline_recovered = decrypt_array(cipher, session_key, nonce)

    rows: list[dict] = []
    baseline = _scenario_row(
        image, cipher, cipher, baseline_recovered,
        crypto=crypto, attack="baseline", parameter="none", parameter_value=0,
    )
    baseline["asymmetric_overhead_bytes"] = overhead
    rows.append(baseline)

    scenarios: list[tuple[str, str, float | int, np.ndarray]] = []
    for i, level in enumerate(noise_levels):
        attacked = salt_pepper_noise(cipher, float(level), seed=seed + 100 + i)
        scenarios.append(("salt_pepper_noise", "fraction", float(level), attacked))
    for i, level in enumerate(block_levels):
        attacked = block_erasure(cipher, float(level), seed=seed + 200 + i)
        scenarios.append(("block_erasure", "area_fraction", float(level), attacked))
    for quality in jpeg_qualities:
        attacked = jpeg_recompress(cipher, int(quality))
        scenarios.append(("jpeg_recompression", "quality", int(quality), attacked))

    if save_images:
        save_png(image, mode, out / "images" / "original.png")
        save_png(cipher, mode, out / "images" / "cipher.png")
        save_png(baseline_recovered, mode, out / "images" / "baseline_recovered.png")

    for idx, (attack, parameter, value, attacked) in enumerate(scenarios, 1):
        recovered = decrypt_array(attacked, session_key, nonce)
        row = _scenario_row(
            image, cipher, attacked, recovered,
            crypto=crypto, attack=attack, parameter=parameter, parameter_value=value,
        )
        row["asymmetric_overhead_bytes"] = overhead
        rows.append(row)
        if save_images:
            safe_value = str(value).replace(".", "p")
            stem = f"{idx:02d}_{attack}_{parameter}_{safe_value}"
            save_png(attacked, mode, out / "images" / f"{stem}.attacked_cipher.png")
            save_png(recovered, mode, out / "images" / f"{stem}.recovered.png")

    _write_csv(out / "robustness.csv", rows)
    _save_robustness_plots(rows, out)

    summary: dict[str, object] = {
        "image": str(Path(image_path).resolve()),
        "crypto": crypto,
        "diffusion_version": "v3",
        "baseline_exact": baseline["exact_match"],
        "scenario_count": len(scenarios),
        "assumption": "attacks are applied to ciphertext before decryption; block_erasure preserves dimensions",
        "groups": {},
    }
    for attack in sorted({r["attack"] for r in rows if r["attack"] != "baseline"}):
        subset = [r for r in rows if r["attack"] == attack]
        group = {}
        for field in ("cipher_npcr_percent", "cipher_uaci_percent", "psnr_recovered_db", "ssim_recovered", "pixel_error_rate_percent"):
            vals = [float(r[field]) for r in subset if math.isfinite(float(r[field]))]
            if vals:
                group[f"{field}_mean"] = statistics.fmean(vals)
                group[f"{field}_min"] = min(vals)
                group[f"{field}_max"] = max(vals)
        summary["groups"][attack] = group

    (out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return {"rows": rows, "summary": summary}


def _flip_session_key_bit(key: bytes, bit_index: int) -> bytes:
    if len(key) != 32:
        raise ValueError("expected 256-bit session key")
    if not 0 <= bit_index < 256:
        raise ValueError("bit_index must be in 0..255")
    out = bytearray(key)
    byte_index = bit_index // 8
    bit_in_byte = bit_index % 8
    out[byte_index] ^= 1 << bit_in_byte
    return bytes(out)


def _save_key_sensitivity_plots(rows: list[dict], out_dir: Path) -> None:
    out = Path(out_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    trials = [int(r["trial"]) for r in rows]
    for field, ylabel, filename in (
        ("cipher_npcr_percent", "Cipher NPCR (%)", "npcr_by_trial.png"),
        ("cipher_uaci_percent", "Cipher UACI (%)", "uaci_by_trial.png"),
        ("wrong_key_ssim", "Wrong-key recovered SSIM", "wrong_key_ssim.png"),
        ("wrong_key_pixel_error_rate_percent", "Wrong-key pixel error rate (%)", "wrong_key_pixel_error.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(trials, [float(r[field]) for r in rows], marker="o")
        ax.set_xlabel("Trial")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)


def run_key_sensitivity(
    image_path: Path,
    out_dir: Path,
    *,
    trials: int = 20,
    seed: int = 42,
    save_images: bool = True,
) -> dict:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    image, mode = load_image(image_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for trial in range(1, trials + 1):
        key = os.urandom(32)
        nonce = os.urandom(16)
        bit_index = int(rng.integers(0, 256))
        changed_key = _flip_session_key_bit(key, bit_index)
        cipher_a = encrypt_array(image, key, nonce)
        cipher_b = encrypt_array(image, changed_key, nonce)
        wrong_recovered = decrypt_array(cipher_a, changed_key, nonce)

        row = {
            "trial": trial,
            "flipped_key_bit": bit_index,
            "cipher_npcr_percent": npcr(cipher_a, cipher_b),
            "cipher_uaci_percent": uaci(cipher_a, cipher_b),
            "wrong_key_exact_match": bool(np.array_equal(image, wrong_recovered)),
            "wrong_key_mse": mse(image, wrong_recovered),
            "wrong_key_psnr_db": psnr(image, wrong_recovered),
            "wrong_key_ssim": ssim(image, wrong_recovered),
            "wrong_key_pixel_error_rate_percent": pixel_error_rate(image, wrong_recovered),
        }
        rows.append(row)
        if trial == 1 and save_images:
            save_png(image, mode, out / "images" / "original.png")
            save_png(cipher_a, mode, out / "images" / "cipher_original_key.png")
            save_png(cipher_b, mode, out / "images" / "cipher_one_bit_key_change.png")
            save_png(wrong_recovered, mode, out / "images" / "wrong_key_recovered.png")

    _write_csv(out / "trials.csv", rows)
    _save_key_sensitivity_plots(rows, out)

    fields = (
        "cipher_npcr_percent",
        "cipher_uaci_percent",
        "wrong_key_mse",
        "wrong_key_psnr_db",
        "wrong_key_ssim",
        "wrong_key_pixel_error_rate_percent",
    )
    summary: dict[str, object] = {
        "image": str(Path(image_path).resolve()),
        "trials": trials,
        "diffusion_version": "v3",
        "wrong_key_exact_match_count": sum(bool(r["wrong_key_exact_match"]) for r in rows),
    }
    for field in fields:
        vals = [float(r[field]) for r in rows if math.isfinite(float(r[field]))]
        if vals:
            summary[f"{field}_mean"] = statistics.fmean(vals)
            summary[f"{field}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            summary[f"{field}_min"] = min(vals)
            summary[f"{field}_max"] = max(vals)
    (out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return {"rows": rows, "summary": summary}
