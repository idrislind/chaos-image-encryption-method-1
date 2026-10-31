import os
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

def generate_keypair(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    priv = X25519PrivateKey.generate()
    priv_path, pub_path = out / "private.pem", out / "public.pem"
    priv_path.write_bytes(priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pub_path.write_bytes(priv.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv_path, pub_path

def _derive(shared, salt):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=b"chaos-image-encryption-v1|x25519-session").derive(shared)

def create_session(public_key_path):
    pub = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    eph = X25519PrivateKey.generate(); salt = os.urandom(16)
    session = _derive(eph.exchange(pub), salt)
    eph_pub = eph.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return session, eph_pub, salt

def recover_session(private_key_path, eph_pub_raw, salt):
    priv = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    eph_pub = X25519PublicKey.from_public_bytes(eph_pub_raw)
    return _derive(priv.exchange(eph_pub), salt)
