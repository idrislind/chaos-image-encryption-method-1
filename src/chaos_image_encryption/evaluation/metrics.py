from __future__ import annotations

import math
from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity


def _as_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError("metrics expect uint8 images")
    if arr.ndim not in (2, 3):
        raise ValueError("metrics expect grayscale or RGB image")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError("RGB image must have exactly 3 channels")
    return arr


def _same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _as_uint8(a)
    b = _as_uint8(b)
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} != {b.shape}")
    return a, b


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _same_shape(a, b)
    delta = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(delta * delta))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value == 0.0:
        return float("inf")
    return float(10.0 * math.log10((255.0 * 255.0) / value))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _same_shape(a, b)
    channel_axis = -1 if a.ndim == 3 else None
    return float(structural_similarity(a, b, data_range=255, channel_axis=channel_axis))


def _entropy_1d(values: np.ndarray) -> float:
    counts = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def entropy(image: np.ndarray) -> Dict[str, float]:
    arr = _as_uint8(image)
    if arr.ndim == 2:
        return {"overall": _entropy_1d(arr)}

    per_channel = {
        "R": _entropy_1d(arr[..., 0]),
        "G": _entropy_1d(arr[..., 1]),
        "B": _entropy_1d(arr[..., 2]),
    }
    per_channel["mean"] = float(np.mean(list(per_channel.values())))
    per_channel["overall"] = _entropy_1d(arr)
    return per_channel


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64).reshape(-1)
    y = y.astype(np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _direction_pairs(channel: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    if direction == "horizontal":
        return channel[:, :-1], channel[:, 1:]
    if direction == "vertical":
        return channel[:-1, :], channel[1:, :]
    if direction == "diagonal":
        return channel[:-1, :-1], channel[1:, 1:]
    raise ValueError(f"unknown direction: {direction}")


def adjacent_correlations(image: np.ndarray) -> Dict[str, float]:
    arr = _as_uint8(image)
    channels = [arr] if arr.ndim == 2 else [arr[..., i] for i in range(3)]
    result: Dict[str, float] = {}
    for direction in ("horizontal", "vertical", "diagonal"):
        vals = []
        for channel in channels:
            a, b = _direction_pairs(channel, direction)
            vals.append(_pearson(a, b))
        result[direction] = float(np.mean(vals))
    return result


def npcr(cipher_a: np.ndarray, cipher_b: np.ndarray) -> float:
    a, b = _same_shape(cipher_a, cipher_b)
    return float(np.mean(a != b) * 100.0)


def uaci(cipher_a: np.ndarray, cipher_b: np.ndarray) -> float:
    a, b = _same_shape(cipher_a, cipher_b)
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.mean(diff / 255.0) * 100.0)


def summarize_triplet(original: np.ndarray, encrypted: np.ndarray, decrypted: np.ndarray) -> dict:
    original, encrypted = _same_shape(original, encrypted)
    original, decrypted = _same_shape(original, decrypted)
    exact = bool(np.array_equal(original, decrypted))
    return {
        "correctness": {
            "exact_match": exact,
            "mse_original_decrypted": mse(original, decrypted),
            "psnr_original_decrypted_db": psnr(original, decrypted),
            "ssim_original_decrypted": ssim(original, decrypted),
        },
        "entropy": {
            "original": entropy(original),
            "encrypted": entropy(encrypted),
            "decrypted": entropy(decrypted),
        },
        "correlation": {
            "original": adjacent_correlations(original),
            "encrypted": adjacent_correlations(encrypted),
            "decrypted": adjacent_correlations(decrypted),
        },
    }
