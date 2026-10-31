from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def _as_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError("attack functions expect uint8 images")
    if arr.ndim not in (2, 3):
        raise ValueError("attack functions expect grayscale or RGB image")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError("RGB image must have exactly 3 channels")
    return arr


def salt_pepper_noise(cipher: np.ndarray, fraction: float, *, seed: int = 0) -> np.ndarray:
    """Corrupt a fraction of ciphertext bytes with salt/pepper values.

    `fraction` is measured over stored uint8 elements. For RGB this means channel
    bytes, not whole pixels. Selected bytes are forced to change so the requested
    corruption level is not silently reduced when a byte already equals 0 or 255.
    """
    arr = _as_uint8(cipher)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    out = arr.copy()
    if fraction == 0.0 or out.size == 0:
        return out

    count = min(out.size, max(1, int(round(fraction * out.size))))
    rng = np.random.default_rng(seed)
    indices = rng.choice(out.size, size=count, replace=False)
    flat = out.reshape(-1)
    choice = rng.integers(0, 2, size=count, dtype=np.uint8)
    values = np.where(choice == 0, np.uint8(0), np.uint8(255)).astype(np.uint8)
    original = flat[indices]
    same = values == original
    values[same] = np.uint8(255) - values[same]
    flat[indices] = values
    return out


def block_erasure(cipher: np.ndarray, area_fraction: float, *, seed: int = 0, fill: int = 0) -> np.ndarray:
    """Erase one contiguous rectangular region while preserving image dimensions.

    This is a cropping/data-loss-like channel attack. True cropping changes the
    ciphertext shape and therefore cannot be passed to the current decryptor. A
    same-shape block erasure isolates the effect of losing a spatial region.
    """
    arr = _as_uint8(cipher)
    if not 0.0 <= area_fraction <= 1.0:
        raise ValueError("area_fraction must be between 0 and 1")
    if not 0 <= int(fill) <= 255:
        raise ValueError("fill must be in 0..255")
    out = arr.copy()
    if area_fraction == 0.0 or out.size == 0:
        return out

    h, w = out.shape[:2]
    scale = float(np.sqrt(area_fraction))
    bh = min(h, max(1, int(round(h * scale))))
    bw = min(w, max(1, int(round(w * scale))))
    rng = np.random.default_rng(seed)
    top = int(rng.integers(0, h - bh + 1)) if h > bh else 0
    left = int(rng.integers(0, w - bw + 1)) if w > bw else 0
    out[top : top + bh, left : left + bw, ...] = np.uint8(fill)
    return out


def jpeg_recompress(cipher: np.ndarray, quality: int) -> np.ndarray:
    """JPEG-encode and decode a ciphertext image, returning the same shape/mode.

    This intentionally applies lossy compression to the encrypted image. The
    returned array can then be fed to the decryptor to measure error propagation.
    """
    arr = _as_uint8(cipher)
    if not 1 <= int(quality) <= 100:
        raise ValueError("quality must be in 1..100")
    mode = "L" if arr.ndim == 2 else "RGB"
    bio = BytesIO()
    Image.fromarray(arr, mode=mode).save(bio, format="JPEG", quality=int(quality), optimize=False)
    bio.seek(0)
    with Image.open(bio) as im:
        decoded = im.convert(mode)
        out = np.array(decoded, dtype=np.uint8)
    if out.shape != arr.shape:
        raise RuntimeError(f"JPEG round-trip changed shape: {arr.shape} -> {out.shape}")
    return out
