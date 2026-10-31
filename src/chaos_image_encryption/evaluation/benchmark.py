from __future__ import annotations

import csv
import json
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import psutil

from ..cipher.core import decrypt_array, encrypt_array
from ..crypto import ecc_backend, rsa_backend
from ..io.image_io import load_image, save_png
from .metrics import adjacent_correlations, entropy, mse, npcr, psnr, ssim, uaci

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".ppm", ".webp", ".avif"
}


@dataclass
class Measurement:
    value: object
    elapsed_ms: float
    start_rss_mib: float
    end_rss_mib: float
    peak_rss_mib: float

    @property
    def peak_rss_delta_mib(self) -> float:
        return max(0.0, self.peak_rss_mib - self.start_rss_mib)


def measure_call(fn: Callable, *args, poll_interval_s: float = 0.002, **kwargs) -> Measurement:
    """Time a call and sample process RSS while it runs.

    RSS is process-level memory and therefore includes the Python interpreter and loaded
    libraries. `peak_rss_delta_mib` is the useful field for per-operation comparisons.
    """
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss
    peak = [start_rss]
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error:
                pass
            stop.wait(poll_interval_s)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    t0 = time.perf_counter()
    try:
        value = fn(*args, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stop.set()
        thread.join(timeout=0.1)
    end_rss = process.memory_info().rss
    peak[0] = max(peak[0], end_rss)
    mib = 1024.0 * 1024.0
    return Measurement(value, elapsed_ms, start_rss / mib, end_rss / mib, peak[0] / mib)


def _json_safe(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if math.isnan(v):
            return "nan"
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def _flip_one_bit(image: np.ndarray) -> np.ndarray:
    changed = image.copy()
    h, w = changed.shape[:2]
    row, col = h // 2, w // 2
    if changed.ndim == 2:
        changed[row, col] ^= np.uint8(1)
    else:
        changed[row, col, 0] ^= np.uint8(1)
    return changed


def _backend_session(crypto: str, public_key: Path):
    if crypto == "rsa":
        key, wrapped = rsa_backend.create_session(public_key)
        material = {"wrapped": wrapped}
        overhead_bytes = len(wrapped)
    elif crypto == "ecc":
        key, eph, salt = ecc_backend.create_session(public_key)
        material = {"eph": eph, "salt": salt}
        overhead_bytes = len(eph) + len(salt)
    else:
        raise ValueError(crypto)
    return key, material, overhead_bytes


def _backend_recover(crypto: str, private_key: Path, material: dict):
    if crypto == "rsa":
        return rsa_backend.recover_session(private_key, material["wrapped"])
    if crypto == "ecc":
        return ecc_backend.recover_session(private_key, material["eph"], material["salt"])
    raise ValueError(crypto)


def benchmark_one(
    image: np.ndarray,
    crypto: str,
    public_key: Path,
    private_key: Path,
    *,
    include_differential: bool = True,
) -> tuple[dict, np.ndarray, np.ndarray]:
    nonce = os.urandom(16)
    session_measure = measure_call(_backend_session, crypto, public_key)
    session_key, material, asymmetric_overhead_bytes = session_measure.value

    enc_measure = measure_call(encrypt_array, image, session_key, nonce)
    cipher = enc_measure.value

    recover_measure = measure_call(_backend_recover, crypto, private_key, material)
    recovered_key = recover_measure.value
    if recovered_key != session_key:
        raise RuntimeError(f"{crypto}: recovered session key differs")

    dec_measure = measure_call(decrypt_array, cipher, recovered_key, nonce)
    decrypted = dec_measure.value

    exact = bool(np.array_equal(image, decrypted))
    image_mib = image.nbytes / (1024.0 * 1024.0)
    corr = adjacent_correlations(cipher)

    row = {
        "crypto": crypto,
        "diffusion_version": "v3",
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "channels": 1 if image.ndim == 2 else int(image.shape[2]),
        "image_bytes": int(image.nbytes),
        "exact_match": exact,
        "mse_decrypted": mse(image, decrypted),
        "psnr_decrypted_db": psnr(image, decrypted),
        "ssim_decrypted": ssim(image, decrypted),
        "entropy_original": entropy(image)["overall"],
        "entropy_encrypted": entropy(cipher)["overall"],
        "corr_h_encrypted": corr["horizontal"],
        "corr_v_encrypted": corr["vertical"],
        "corr_d_encrypted": corr["diagonal"],
        "session_create_ms": session_measure.elapsed_ms,
        "session_recover_ms": recover_measure.elapsed_ms,
        "image_encrypt_ms": enc_measure.elapsed_ms,
        "image_decrypt_ms": dec_measure.elapsed_ms,
        "total_encrypt_ms": session_measure.elapsed_ms + enc_measure.elapsed_ms,
        "total_decrypt_ms": recover_measure.elapsed_ms + dec_measure.elapsed_ms,
        "encrypt_throughput_mib_s": image_mib / (enc_measure.elapsed_ms / 1000.0) if enc_measure.elapsed_ms else float("inf"),
        "decrypt_throughput_mib_s": image_mib / (dec_measure.elapsed_ms / 1000.0) if dec_measure.elapsed_ms else float("inf"),
        "encrypt_peak_rss_delta_mib": enc_measure.peak_rss_delta_mib,
        "decrypt_peak_rss_delta_mib": dec_measure.peak_rss_delta_mib,
        "asymmetric_overhead_bytes": int(asymmetric_overhead_bytes),
    }

    if include_differential:
        changed = _flip_one_bit(image)
        # Keep the same key and nonce so only the one-bit plaintext change is measured.
        c1 = encrypt_array(image, session_key, nonce)
        c2 = encrypt_array(changed, session_key, nonce)
        row["npcr_percent"] = npcr(c1, c2)
        row["uaci_percent"] = uaci(c1, c2)
    else:
        row["npcr_percent"] = float("nan")
        row["uaci_percent"] = float("nan")

    return row, cipher, decrypted


def generate_benchmark_keys(out_dir: Path) -> tuple[dict, dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    rows = {}
    for crypto, backend in (("rsa", rsa_backend), ("ecc", ecc_backend)):
        key_dir = out_dir / crypto
        measurement = measure_call(backend.generate_keypair, key_dir)
        private_key, public_key = measurement.value
        paths[crypto] = {"private": Path(private_key), "public": Path(public_key)}
        rows[crypto] = {
            "crypto": crypto,
            "keygen_ms": measurement.elapsed_ms,
            "private_key_file_bytes": Path(private_key).stat().st_size,
            "public_key_file_bytes": Path(public_key).stat().st_size,
        }
    return paths, rows


def discover_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k, "")) for k in fields})


def _aggregate(rows: list[dict]) -> list[dict]:
    metric_fields = [
        "entropy_encrypted", "corr_h_encrypted", "corr_v_encrypted", "corr_d_encrypted",
        "npcr_percent", "uaci_percent", "session_create_ms", "session_recover_ms",
        "image_encrypt_ms", "image_decrypt_ms", "total_encrypt_ms", "total_decrypt_ms",
        "encrypt_throughput_mib_s", "decrypt_throughput_mib_s",
        "encrypt_peak_rss_delta_mib", "decrypt_peak_rss_delta_mib",
    ]
    summaries = []
    for crypto in sorted({r["crypto"] for r in rows}):
        subset = [r for r in rows if r["crypto"] == crypto]
        summary = {
            "crypto": crypto,
            "images": len(subset),
            "exact_match_rate_percent": 100.0 * sum(bool(r["exact_match"]) for r in subset) / len(subset),
        }
        for field in metric_fields:
            vals = []
            for r in subset:
                try:
                    value = float(r[field])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    vals.append(value)
            if vals:
                summary[f"{field}_mean"] = statistics.fmean(vals)
                summary[f"{field}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                summary[f"{field}_min"] = min(vals)
                summary[f"{field}_max"] = max(vals)
        summaries.append(summary)
    return summaries


def run_batch(
    input_dir: Path,
    out_dir: Path,
    *,
    crypto: str = "all",
    limit: int | None = None,
    recursive: bool = False,
    include_differential: bool = True,
    save_images: bool = False,
) -> dict:
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = discover_images(input_dir, recursive=recursive)
    if limit is not None:
        images = images[: max(0, limit)]
    if not images:
        raise ValueError(f"no supported images found in {input_dir}")

    key_paths, keygen_rows = generate_benchmark_keys(out_dir / "keys")
    selected = ("rsa", "ecc") if crypto == "all" else (crypto,)
    rows = []

    for index, path in enumerate(images, 1):
        image, mode = load_image(path)
        relative = path.relative_to(input_dir).as_posix()
        print(f"[{index}/{len(images)}] {relative}")
        for backend in selected:
            row, cipher, decrypted = benchmark_one(
                image,
                backend,
                key_paths[backend]["public"],
                key_paths[backend]["private"],
                include_differential=include_differential,
            )
            row = {"image": relative, "mode": mode, **row}
            rows.append(row)
            print(
                f"  {backend.upper():3s} exact={row['exact_match']} "
                f"enc={row['image_encrypt_ms']:.2f}ms dec={row['image_decrypt_ms']:.2f}ms "
                f"NPCR={row['npcr_percent']:.3f}% UACI={row['uaci_percent']:.3f}%"
            )
            if save_images:
                stem = f"{index:04d}_{path.stem}_{backend}"
                save_png(cipher, mode, out_dir / "images" / f"{stem}.enc.png")
                save_png(decrypted, mode, out_dir / "images" / f"{stem}.dec.png")

    summaries = _aggregate(rows)
    _write_csv(out_dir / "per_image.csv", rows)
    _write_csv(out_dir / "summary.csv", summaries)
    _write_csv(out_dir / "keygen.csv", [keygen_rows[c] for c in selected])
    payload = {
        "input_dir": str(input_dir.resolve()),
        "image_count": len(images),
        "crypto": list(selected),
        "differential_enabled": include_differential,
        "rows": [_json_safe(r) for r in rows],
        "summary": [_json_safe(r) for r in summaries],
        "keygen": [_json_safe(keygen_rows[c]) for c in selected],
    }
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_single(image_path: Path, out_dir: Path, include_differential: bool = True, save_images: bool = True) -> dict:
    image, mode = load_image(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_paths, keygen_rows = generate_benchmark_keys(out_dir / "keys")
    rows = []
    for backend in ("rsa", "ecc"):
        row, cipher, decrypted = benchmark_one(
            image, backend, key_paths[backend]["public"], key_paths[backend]["private"],
            include_differential=include_differential,
        )
        row = {"image": Path(image_path).name, "mode": mode, **row}
        rows.append(row)
        if save_images:
            save_png(cipher, mode, out_dir / "images" / f"{backend}.enc.png")
            save_png(decrypted, mode, out_dir / "images" / f"{backend}.dec.png")
    summaries = _aggregate(rows)
    _write_csv(out_dir / "rsa_vs_ecc.csv", rows)
    _write_csv(out_dir / "summary.csv", summaries)
    _write_csv(out_dir / "keygen.csv", [keygen_rows["rsa"], keygen_rows["ecc"]])
    payload = {"rows": [_json_safe(r) for r in rows], "summary": [_json_safe(r) for r in summaries], "keygen": list(keygen_rows.values())}
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
