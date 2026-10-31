from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class HenonConfig:
    a: float = 1.4
    b: float = 0.3
    burn_in: int = 1000

class HenonDivergenceError(RuntimeError):
    pass

def generate_henon(x0: float, y0: float, length: int, config: HenonConfig = HenonConfig()):
    if length < 0:
        raise ValueError("length must be >= 0")
    x, y = float(x0), float(y0)
    xs = np.empty(length, dtype=np.float64)
    ys = np.empty(length, dtype=np.float64)
    out_i = 0
    for i in range(config.burn_in + length):
        x, y = 1.0 - config.a * x * x + y, config.b * x
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 1e6 or abs(y) > 1e6:
            raise HenonDivergenceError("Hénon orbit diverged")
        if i >= config.burn_in:
            xs[out_i], ys[out_i] = x, y
            out_i += 1
    return xs, ys

def quantize_to_uint8(sequence: np.ndarray) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float64)
    if not np.all(np.isfinite(seq)):
        raise ValueError("sequence contains NaN or Inf")
    vals = np.floor(np.abs(seq) * 1e14).astype(np.uint64)
    return (vals & np.uint64(255)).astype(np.uint8)
