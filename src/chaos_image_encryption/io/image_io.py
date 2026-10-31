from pathlib import Path
import numpy as np
from PIL import Image


def load_image(path):
    """Load an image as uint8 grayscale (L) or RGB.

    Palette, RGBA, CMYK, and other non-grayscale modes are normalized to RGB
    so experimental metrics operate on a consistent pixel representation.
    """
    with Image.open(path) as im:
        if im.mode == "L":
            converted = im.copy()
            mode = "L"
        else:
            converted = im.convert("RGB")
            mode = "RGB"
        return np.array(converted, dtype=np.uint8), mode


def save_png(array, mode, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode=mode).save(out, format="PNG")
