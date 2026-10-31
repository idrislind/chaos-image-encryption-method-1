from __future__ import annotations
import struct
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

def _shape_info(shape):
    return b"".join(struct.pack(">I", int(v)) for v in shape)

def derive_bytes(session_key: bytes, nonce: bytes, shape, domain: bytes, length: int) -> bytes:
    if len(session_key) != 32 or len(nonce) != 16:
        raise ValueError("session_key must be 32 bytes and nonce 16 bytes")
    info = b"chaos-image-encryption-v1|" + domain + b"|" + _shape_info(shape)
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=nonce, info=info).derive(session_key)

def derive_henon_seed(session_key: bytes, nonce: bytes, shape, domain: bytes, attempt: int = 0):
    m = derive_bytes(session_key, nonce, shape, domain + b"|attempt=" + str(attempt).encode(), 16)
    denom = float((1 << 64) - 1)
    ux = int.from_bytes(m[:8], "big") / denom
    uy = int.from_bytes(m[8:], "big") / denom
    return -0.20 + 0.40 * ux, -0.20 + 0.40 * uy

def derive_iv_bytes(session_key: bytes, nonce: bytes, shape):
    m = derive_bytes(session_key, nonce, shape, b"diffusion-ivs", 2)
    return m[0], m[1]
