from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chaos_image_encryption.cipher.core import encrypt_array, decrypt_array
from chaos_image_encryption.crypto import rsa_backend, ecc_backend
from chaos_image_encryption.evaluation.metrics import npcr, uaci
from chaos_image_encryption.evaluation.report import write_triplet_report
from chaos_image_encryption.evaluation.benchmark import run_batch, run_single
from chaos_image_encryption.evaluation.robustness import run_key_sensitivity, run_robustness
from chaos_image_encryption.io.image_io import load_image, save_png


def e(b):
    return base64.b64encode(b).decode("ascii")


def d(s):
    return base64.b64decode(s.encode("ascii"))


def meta_path(p: Path):
    return p.with_name((p.name[:-4] if p.name.endswith(".png") else p.name) + ".json")


def keygen(a):
    fn = rsa_backend.generate_keypair if a.type == "rsa" else ecc_backend.generate_keypair
    priv, pub = fn(a.out_dir)
    print(f"private: {priv}\npublic : {pub}")


def encrypt(a):
    image, mode = load_image(a.image)
    nonce = os.urandom(16)
    if a.crypto == "rsa":
        key, wrapped = rsa_backend.create_session(a.public_key)
        asym = {"backend": "rsa-3072-oaep-sha256", "wrapped_session_key_b64": e(wrapped)}
    else:
        key, eph, salt = ecc_backend.create_session(a.public_key)
        asym = {
            "backend": "x25519-hkdf-sha256",
            "ephemeral_public_key_b64": e(eph),
            "ecdh_salt_b64": e(salt),
        }

    cipher = encrypt_array(image, key, nonce)
    out = Path(a.out)
    save_png(cipher, mode, out)
    meta = {
        "version": "1.3",
        "method": "henon-2d-permutation-two-round-nonlinear-diffusion",
        "diffusion_version": "v3",
        "mode": mode,
        "shape": list(image.shape),
        "dtype": "uint8",
        "nonce_b64": e(nonce),
        "henon": {"a": 1.4, "b": 0.3, "burn_in": 1000},
        "asymmetric": asym,
    }
    mp = meta_path(out)
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cipher   : {out}\nmetadata : {mp}")


def decrypt(a):
    cipher, mode = load_image(a.image)
    meta = json.loads(Path(a.metadata).read_text(encoding="utf-8"))
    if meta["mode"] != mode or list(cipher.shape) != meta["shape"]:
        raise ValueError("cipher image does not match metadata")

    asym = meta["asymmetric"]
    nonce = d(meta["nonce_b64"])
    if asym["backend"].startswith("rsa-"):
        key = rsa_backend.recover_session(a.private_key, d(asym["wrapped_session_key_b64"]))
    else:
        key = ecc_backend.recover_session(
            a.private_key,
            d(asym["ephemeral_public_key_b64"]),
            d(asym["ecdh_salt_b64"]),
        )

    diffusion_version = meta.get("diffusion_version", "v1")
    plain = decrypt_array(cipher, key, nonce, diffusion_version=diffusion_version)
    save_png(plain, mode, a.out)
    print(f"decrypted: {a.out}")


def evaluate(a):
    original, original_mode = load_image(a.original)
    encrypted, encrypted_mode = load_image(a.encrypted)
    decrypted, decrypted_mode = load_image(a.decrypted)
    if not (original_mode == encrypted_mode == decrypted_mode):
        raise ValueError(
            f"image modes differ: original={original_mode}, encrypted={encrypted_mode}, decrypted={decrypted_mode}"
        )

    metrics = write_triplet_report(original, encrypted, decrypted, a.out_dir)
    c = metrics["correctness"]
    ent = metrics["entropy"]
    corr = metrics["correlation"]

    psnr_value = c["psnr_original_decrypted_db"]
    psnr_text = "inf" if np.isinf(psnr_value) else f"{psnr_value:.6f} dB"
    print(f"exact match          : {c['exact_match']}")
    print(f"MSE                  : {c['mse_original_decrypted']:.6f}")
    print(f"PSNR                 : {psnr_text}")
    print(f"SSIM                 : {c['ssim_original_decrypted']:.6f}")
    print(f"entropy original     : {ent['original']['overall']:.6f}")
    print(f"entropy encrypted    : {ent['encrypted']['overall']:.6f}")
    print(f"corr H encrypted     : {corr['encrypted']['horizontal']:.6f}")
    print(f"corr V encrypted     : {corr['encrypted']['vertical']:.6f}")
    print(f"corr D encrypted     : {corr['encrypted']['diagonal']:.6f}")
    print(f"report               : {Path(a.out_dir).resolve()}")


def differential(a):
    image, mode = load_image(a.image)
    changed = image.copy()
    h, w = changed.shape[:2]
    row = h // 2 if a.row is None else a.row
    col = w // 2 if a.col is None else a.col
    if not (0 <= row < h and 0 <= col < w):
        raise ValueError(f"pixel ({row}, {col}) outside image shape {changed.shape}")

    if changed.ndim == 2:
        changed[row, col] ^= np.uint8(1)
    else:
        channel = a.channel
        if not 0 <= channel < 3:
            raise ValueError("--channel must be 0, 1, or 2 for RGB")
        changed[row, col, channel] ^= np.uint8(1)

    # Same 256-bit key and nonce are intentionally used for both encryptions so
    # NPCR/UACI isolate the effect of a one-bit plaintext change.
    session_key = os.urandom(32)
    nonce = os.urandom(16)
    cipher_a = encrypt_array(image, session_key, nonce)
    cipher_b = encrypt_array(changed, session_key, nonce)

    npcr_value = npcr(cipher_a, cipher_b)
    uaci_value = uaci(cipher_a, cipher_b)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_png(image, mode, out / "original.png")
    save_png(changed, mode, out / "modified_one_bit.png")
    save_png(cipher_a, mode, out / "cipher_original.png")
    save_png(cipher_b, mode, out / "cipher_modified.png")

    result = {
        "changed_pixel": {"row": row, "col": col, "channel": None if changed.ndim == 2 else a.channel},
        "change": "flip least-significant bit (XOR 1)",
        "same_session_key": True,
        "same_nonce": True,
        "npcr_percent": npcr_value,
        "uaci_percent": uaci_value,
    }
    (out / "differential_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"NPCR : {npcr_value:.6f}%")
    print(f"UACI : {uaci_value:.6f}%")
    print(f"report: {out.resolve()}")



def differential_sweep(a):
    image, _ = load_image(a.image)
    changed = image.copy()
    h, w = changed.shape[:2]
    row = h // 2
    col = w // 2
    if changed.ndim == 2:
        changed[row, col] ^= np.uint8(1)
        channel = None
    else:
        channel = a.channel
        if not 0 <= channel < 3:
            raise ValueError("--channel must be 0, 1, or 2 for RGB")
        changed[row, col, channel] ^= np.uint8(1)

    rows = []
    for i in range(1, a.trials + 1):
        key = os.urandom(32)
        nonce = os.urandom(16)
        c1 = encrypt_array(image, key, nonce)
        c2 = encrypt_array(changed, key, nonce)
        n = npcr(c1, c2)
        u = uaci(c1, c2)
        rows.append({"trial": i, "npcr_percent": n, "uaci_percent": u})
        print(f"[{i}/{a.trials}] NPCR={n:.3f}% UACI={u:.3f}%")

    npcrs = [r["npcr_percent"] for r in rows]
    uacis = [r["uaci_percent"] for r in rows]
    summary = {
        "trials": a.trials,
        "changed_pixel": {"row": row, "col": col, "channel": channel},
        "change": "flip least-significant bit (XOR 1)",
        "diffusion_version": "v3",
        "npcr_percent_mean": statistics.fmean(npcrs),
        "npcr_percent_std": statistics.pstdev(npcrs) if len(npcrs) > 1 else 0.0,
        "npcr_percent_min": min(npcrs),
        "npcr_percent_max": max(npcrs),
        "uaci_percent_mean": statistics.fmean(uacis),
        "uaci_percent_std": statistics.pstdev(uacis) if len(uacis) > 1 else 0.0,
        "uaci_percent_min": min(uacis),
        "uaci_percent_max": max(uacis),
    }

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "npcr_percent", "uaci_percent"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDifferential sweep summary")
    print(f"NPCR mean/std/min/max : {summary['npcr_percent_mean']:.3f} / {summary['npcr_percent_std']:.3f} / {summary['npcr_percent_min']:.3f} / {summary['npcr_percent_max']:.3f}%")
    print(f"UACI mean/std/min/max : {summary['uaci_percent_mean']:.3f} / {summary['uaci_percent_std']:.3f} / {summary['uaci_percent_min']:.3f} / {summary['uaci_percent_max']:.3f}%")
    print(f"report                : {out.resolve()}")


def benchmark(a):
    payload = run_single(
        a.image,
        a.out_dir,
        include_differential=not a.skip_differential,
        save_images=not a.no_save_images,
    )
    print("\nRSA vs ECC benchmark")
    for row in payload["rows"]:
        print(
            f"{row['crypto'].upper():3s} | exact={row['exact_match']} | "
            f"enc={row['image_encrypt_ms']:.2f} ms | dec={row['image_decrypt_ms']:.2f} ms | "
            f"throughput enc/dec={row['encrypt_throughput_mib_s']:.2f}/{row['decrypt_throughput_mib_s']:.2f} MiB/s | "
            f"NPCR={row['npcr_percent']:.3f}% | UACI={row['uaci_percent']:.3f}%"
        )
    print(f"report: {Path(a.out_dir).resolve()}")


def batch_benchmark(a):
    payload = run_batch(
        a.input_dir,
        a.out_dir,
        crypto=a.crypto,
        limit=a.limit,
        recursive=a.recursive,
        include_differential=not a.skip_differential,
        save_images=a.save_images,
    )
    print("\nAggregate summary")
    for row in payload["summary"]:
        print(
            f"{row['crypto'].upper():3s} | images={row['images']} | "
            f"exact={row['exact_match_rate_percent']:.1f}% | "
            f"NPCR mean={row.get('npcr_percent_mean', float('nan')):.3f}% | "
            f"UACI mean={row.get('uaci_percent_mean', float('nan')):.3f}% | "
            f"enc mean={row.get('image_encrypt_ms_mean', float('nan')):.2f} ms | "
            f"dec mean={row.get('image_decrypt_ms_mean', float('nan')):.2f} ms"
        )
    print(f"report: {Path(a.out_dir).resolve()}")



def robustness(a):
    payload = run_robustness(
        a.image,
        a.out_dir,
        crypto=a.crypto,
        noise_levels=tuple(a.noise_levels),
        block_levels=tuple(a.block_levels),
        jpeg_qualities=tuple(a.jpeg_qualities),
        seed=a.seed,
        save_images=not a.no_save_images,
    )
    print("\nRobustness / ciphertext attack experiment")
    for row in payload["rows"]:
        psnr_value = row["psnr_recovered_db"]
        psnr_text = "inf" if np.isinf(psnr_value) else f"{psnr_value:.2f}dB"
        print(
            f"{row['attack']:18s} {row['parameter']}={row['parameter_value']} | "
            f"cipher-change={row['cipher_npcr_percent']:.3f}% | "
            f"PSNR={psnr_text} SSIM={row['ssim_recovered']:.4f} "
            f"pixel-error={row['pixel_error_rate_percent']:.3f}%"
        )
    print(f"report: {Path(a.out_dir).resolve()}")


def key_sensitivity(a):
    payload = run_key_sensitivity(
        a.image,
        a.out_dir,
        trials=a.trials,
        seed=a.seed,
        save_images=not a.no_save_images,
    )
    summary = payload["summary"]
    print("\nKey sensitivity summary")
    print(
        f"Cipher NPCR mean/std : {summary['cipher_npcr_percent_mean']:.3f} / "
        f"{summary['cipher_npcr_percent_std']:.3f}%"
    )
    print(
        f"Cipher UACI mean/std : {summary['cipher_uaci_percent_mean']:.3f} / "
        f"{summary['cipher_uaci_percent_std']:.3f}%"
    )
    print(
        f"Wrong-key SSIM mean  : {summary['wrong_key_ssim_mean']:.6f}\n"
        f"Wrong-key pixel err  : {summary['wrong_key_pixel_error_rate_percent_mean']:.3f}%\n"
        f"Wrong-key exact hits : {summary['wrong_key_exact_match_count']}/{summary['trials']}"
    )
    print(f"report: {Path(a.out_dir).resolve()}")

def parser():
    p = argparse.ArgumentParser(description="Method 1: asymmetric + 2D Hénon image encryption")
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("keygen")
    q.add_argument("--type", choices=["rsa", "ecc"], required=True)
    q.add_argument("--out-dir", required=True)
    q.set_defaults(fn=keygen)

    q = s.add_parser("encrypt")
    q.add_argument("--image", required=True)
    q.add_argument("--crypto", choices=["rsa", "ecc"], required=True)
    q.add_argument("--public-key", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=encrypt)

    q = s.add_parser("decrypt")
    q.add_argument("--image", required=True)
    q.add_argument("--metadata", required=True)
    q.add_argument("--private-key", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=decrypt)

    q = s.add_parser("evaluate", help="Evaluate original/encrypted/decrypted image triplet")
    q.add_argument("--original", required=True)
    q.add_argument("--encrypted", required=True)
    q.add_argument("--decrypted", required=True)
    q.add_argument("--out-dir", default="results/evaluation")
    q.set_defaults(fn=evaluate)

    q = s.add_parser("differential", help="Run one-bit plaintext NPCR/UACI experiment")
    q.add_argument("--image", required=True)
    q.add_argument("--out-dir", default="results/differential")
    q.add_argument("--row", type=int)
    q.add_argument("--col", type=int)
    q.add_argument("--channel", type=int, default=0)
    q.set_defaults(fn=differential)

    q = s.add_parser("differential-sweep", help="Repeat one-bit NPCR/UACI test across random key/nonce trials")
    q.add_argument("--image", required=True)
    q.add_argument("--trials", type=int, default=10)
    q.add_argument("--channel", type=int, default=0)
    q.add_argument("--out-dir", default="results/differential_sweep")
    q.set_defaults(fn=differential_sweep)



    q = s.add_parser("robustness", help="Attack ciphertext with noise, block erasure, and JPEG recompression")
    q.add_argument("--image", required=True)
    q.add_argument("--crypto", choices=["rsa", "ecc"], default="rsa")
    q.add_argument("--out-dir", default="results/robustness")
    q.add_argument("--noise-levels", nargs="+", type=float, default=[0.001, 0.005, 0.01])
    q.add_argument("--block-levels", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25])
    q.add_argument("--jpeg-qualities", nargs="+", type=int, default=[90, 70, 50])
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--no-save-images", action="store_true")
    q.set_defaults(fn=robustness)

    q = s.add_parser("key-sensitivity", help="Flip one session-key bit and measure ciphertext/wrong-key sensitivity")
    q.add_argument("--image", required=True)
    q.add_argument("--trials", type=int, default=20)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--out-dir", default="results/key_sensitivity")
    q.add_argument("--no-save-images", action="store_true")
    q.set_defaults(fn=key_sensitivity)

    q = s.add_parser("benchmark", help="Benchmark RSA vs ECC on one image")
    q.add_argument("--image", required=True)
    q.add_argument("--out-dir", default="results/benchmark")
    q.add_argument("--skip-differential", action="store_true", help="Skip NPCR/UACI to run faster")
    q.add_argument("--no-save-images", action="store_true", help="Do not save encrypted/decrypted benchmark images")
    q.set_defaults(fn=benchmark)

    q = s.add_parser("batch-benchmark", help="Benchmark a folder of images and aggregate RSA/ECC results")
    q.add_argument("--input-dir", required=True)
    q.add_argument("--out-dir", default="results/batch")
    q.add_argument("--crypto", choices=["all", "rsa", "ecc"], default="all")
    q.add_argument("--limit", type=int, help="Only process the first N discovered images")
    q.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    q.add_argument("--skip-differential", action="store_true", help="Skip per-image NPCR/UACI")
    q.add_argument("--save-images", action="store_true", help="Save encrypted/decrypted outputs for every image")
    q.set_defaults(fn=batch_benchmark)

    return p


if __name__ == "__main__":
    a = parser().parse_args()
    a.fn(a)
