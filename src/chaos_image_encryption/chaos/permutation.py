import numpy as np

def permutation_from_sequence(sequence):
    s = np.asarray(sequence)
    if s.ndim != 1:
        raise ValueError("sequence must be 1-D")
    return np.argsort(s, kind="mergesort")

def permute_pixels(image, perm):
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    pixels = arr.reshape(h * w, channels)
    if len(perm) != len(pixels):
        raise ValueError("permutation length mismatch")
    return pixels[perm].reshape(arr.shape)

def inverse_permute_pixels(image, perm):
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    pixels = arr.reshape(h * w, channels)
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm), dtype=perm.dtype)
    return pixels[inverse].reshape(arr.shape)
