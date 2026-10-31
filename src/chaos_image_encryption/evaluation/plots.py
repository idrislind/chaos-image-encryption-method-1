from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_histogram(image: np.ndarray, path, title: str):
    arr = np.asarray(image, dtype=np.uint8)
    path = _prepare(path)
    plt.figure(figsize=(8, 5))
    if arr.ndim == 2:
        plt.hist(arr.ravel(), bins=256, range=(0, 256), histtype="step")
    else:
        labels = ("R", "G", "B")
        for idx, label in enumerate(labels):
            plt.hist(arr[..., idx].ravel(), bins=256, range=(0, 256), histtype="step", label=label)
        plt.legend()
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _pairs(channel: np.ndarray, direction: str):
    if direction == "horizontal":
        return channel[:, :-1].ravel(), channel[:, 1:].ravel()
    if direction == "vertical":
        return channel[:-1, :].ravel(), channel[1:, :].ravel()
    if direction == "diagonal":
        return channel[:-1, :-1].ravel(), channel[1:, 1:].ravel()
    raise ValueError(direction)


def save_correlation_scatter(image: np.ndarray, path, title: str, direction: str, max_points: int = 5000):
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3:
        # Use luminance-like average only for visualization; numeric metrics are channel-averaged separately.
        arr = np.rint(arr.astype(np.float64).mean(axis=2)).astype(np.uint8)
    x, y = _pairs(arr, direction)
    if x.size > max_points:
        # deterministic evenly spaced sample for reproducible figures
        idx = np.linspace(0, x.size - 1, max_points, dtype=np.int64)
        x, y = x[idx], y[idx]
    path = _prepare(path)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=4, alpha=0.25)
    plt.title(title)
    plt.xlabel("Pixel i")
    plt.ylabel("Adjacent pixel")
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
