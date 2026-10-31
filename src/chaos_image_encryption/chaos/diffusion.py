from __future__ import annotations

import numpy as np

# AES S-box is used here only as a fixed, well-studied 8-bit bijective nonlinear
# substitution layer. This project does NOT claim that using this S-box makes the
# experimental chaos cipher equivalent to AES.
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)

INV_SBOX = np.empty(256, dtype=np.uint8)
INV_SBOX[SBOX] = np.arange(256, dtype=np.uint8)


def _validate(data, k1, k2):
    p = np.asarray(data, dtype=np.uint8).reshape(-1)
    a = np.asarray(k1, dtype=np.uint8).reshape(-1)
    b = np.asarray(k2, dtype=np.uint8).reshape(-1)
    if not (len(p) == len(a) == len(b)):
        raise ValueError("length mismatch")
    return p, a, b


def _derived_streams(k1: np.ndarray, k2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive nonlinear additive streams and odd multiplicative streams.

    The multipliers are forced odd so every multiplication is invertible modulo 256.
    Everything is deterministic from the two Hénon-derived byte streams.
    """
    n = len(k1)
    if n == 0:
        return k1.copy(), k2.copy(), k1.copy(), k2.copy()
    idx = np.arange(n, dtype=np.uint64)
    r2 = np.roll(k2, 1)
    r1 = np.roll(k1, -1)
    t3 = ((k1.astype(np.uint16) + r2.astype(np.uint16) + (idx & 0xFF).astype(np.uint16)) & 0xFF).astype(np.uint8)
    t4 = ((k2.astype(np.uint16) + r1.astype(np.uint16) + ((idx * 29 + 71) & 0xFF).astype(np.uint16)) & 0xFF).astype(np.uint8)
    k3 = SBOX[t3]
    k4 = SBOX[t4]
    a = (SBOX[np.bitwise_xor(k1, k3)] | np.uint8(1)).astype(np.uint8)
    b = (SBOX[np.bitwise_xor(k2, k4)] | np.uint8(1)).astype(np.uint8)
    return k3, k4, a, b


# Multiplicative inverses for every odd byte modulo 256. Even entries remain 0
# and are never used because the recurrence multipliers are always odd.
INV_ODD = np.zeros(256, dtype=np.uint16)
for _a in range(1, 256, 2):
    INV_ODD[_a] = pow(_a, -1, 256)


def _affine_feedback_forward(x: np.ndarray, add_key: np.ndarray, mult: np.ndarray, iv: int) -> np.ndarray:
    """Vectorized recurrence y[i] = mult[i]*y[i-1] + x[i] + add_key[i] (mod 256)."""
    if len(x) == 0:
        return x.copy()
    z = ((x.astype(np.uint16) + add_key.astype(np.uint16)) & 0xFF).astype(np.uint8)

    # Prefix products are valid modulo 256 by taking the low byte after uint64
    # accumulation; uint64 wrap does not affect those low 8 bits.
    prefix_mult = (np.multiply.accumulate(mult.astype(np.uint64)) & np.uint64(255)).astype(np.uint8)
    inv_prefix = INV_ODD[prefix_mult].astype(np.uint64)
    weighted = z.astype(np.uint64) * inv_prefix
    prefix_sum = np.cumsum(weighted, dtype=np.uint64) & np.uint64(255)
    return (
        prefix_mult.astype(np.uint64)
        * ((np.uint64(iv) + prefix_sum) & np.uint64(255))
        & np.uint64(255)
    ).astype(np.uint8)


def _affine_feedback_inverse(y: np.ndarray, add_key: np.ndarray, mult: np.ndarray, iv: int) -> np.ndarray:
    if len(y) == 0:
        return y.copy()
    prev = np.empty(len(y), dtype=np.uint16)
    prev[0], prev[1:] = iv, y[:-1]
    z = ((y.astype(np.int32) - mult.astype(np.int32) * prev.astype(np.int32)) % 256).astype(np.uint8)
    return ((z.astype(np.int32) - add_key.astype(np.int32)) % 256).astype(np.uint8)


def diffuse(data, k1, k2, iv1: int, iv2: int):
    """Milestone 3.5 nonlinear affine two-pass diffusion (v2).

    Compared with v1, the feedback multiplier changes at every byte and is derived
    from the chaotic keystream. The odd multipliers make the transform reversible
    modulo 256, while the S-box layers break the constant-delta behaviour observed
    in the v1 differential experiment.
    """
    p, k1, k2 = _validate(data, k1, k2)
    if len(p) == 0:
        return p.copy()

    k3, k4, a, b = _derived_streams(k1, k2)
    q = SBOX[np.bitwise_xor(p, k1)]
    f = _affine_feedback_forward(q, k2, a, iv1)
    m = SBOX[np.bitwise_xor(f, k3)]
    return _affine_feedback_forward(m[::-1], k4[::-1], b[::-1], iv2)[::-1]


def inverse_diffuse(cipher, k1, k2, iv1: int, iv2: int):
    c, k1, k2 = _validate(cipher, k1, k2)
    if len(c) == 0:
        return c.copy()

    k3, k4, a, b = _derived_streams(k1, k2)
    m = _affine_feedback_inverse(c[::-1], k4[::-1], b[::-1], iv2)[::-1]
    f = np.bitwise_xor(INV_SBOX[m], k3)
    q = _affine_feedback_inverse(f, k2, a, iv1)
    return np.bitwise_xor(INV_SBOX[q], k1).astype(np.uint8)


# Legacy Milestone 3 diffusion retained for reproducibility/backward compatibility.
def diffuse_v1(data, k1, k2, iv1: int, iv2: int):
    p, k1, k2 = _validate(data, k1, k2)
    f = (np.uint64(iv1) + np.cumsum(p.astype(np.uint16) + k1.astype(np.uint16), dtype=np.uint64)) & np.uint64(255)
    f = f.astype(np.uint8)
    c_rev = (np.uint64(iv2) + np.cumsum((f.astype(np.uint16) + k2.astype(np.uint16))[::-1], dtype=np.uint64)) & np.uint64(255)
    return c_rev[::-1].astype(np.uint8)


def inverse_diffuse_v1(cipher, k1, k2, iv1: int, iv2: int):
    c, k1, k2 = _validate(cipher, k1, k2)
    c_next = np.empty(len(c), dtype=np.uint16)
    if len(c):
        c_next[:-1], c_next[-1] = c[1:], iv2
    f = ((c.astype(np.int32) - k2.astype(np.int32) - c_next.astype(np.int32)) % 256).astype(np.uint8)
    f_prev = np.empty(len(f), dtype=np.uint16)
    if len(f):
        f_prev[0], f_prev[1:] = iv1, f[:-1]
    return ((f.astype(np.int32) - k1.astype(np.int32) - f_prev.astype(np.int32)) % 256).astype(np.uint8)


diffuse_v2 = diffuse
inverse_diffuse_v2 = inverse_diffuse


# v3: two independently mixed nonlinear rounds to suppress rare differential weak cases.
def _round2_material(k1: np.ndarray, k2: np.ndarray, iv1: int, iv2: int):
    # Domain-separate the second round from the first using nonlinear byte permutations.
    r1 = SBOX[np.bitwise_xor(np.roll(k2, 7), np.uint8(0xA5))]
    r2 = SBOX[np.bitwise_xor(np.roll(k1, -11), np.uint8(0x5A))]
    riv1 = int(SBOX[np.uint8(iv2 ^ 0x3C)])
    riv2 = int(SBOX[np.uint8(iv1 ^ 0xC3)])
    return r1.astype(np.uint8), r2.astype(np.uint8), riv1, riv2

def diffuse_v3(data, k1, k2, iv1: int, iv2: int):
    p, k1, k2 = _validate(data, k1, k2)
    c1 = diffuse(p, k1, k2, iv1, iv2)
    r1, r2, riv1, riv2 = _round2_material(k1, k2, iv1, iv2)
    return diffuse(c1, r1, r2, riv1, riv2)

def inverse_diffuse_v3(cipher, k1, k2, iv1: int, iv2: int):
    c, k1, k2 = _validate(cipher, k1, k2)
    r1, r2, riv1, riv2 = _round2_material(k1, k2, iv1, iv2)
    p1 = inverse_diffuse(c, r1, r2, riv1, riv2)
    return inverse_diffuse(p1, k1, k2, iv1, iv2)
