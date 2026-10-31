import os
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def generate_keypair(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    priv = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    priv_path, pub_path = out / "private.pem", out / "public.pem"
    priv_path.write_bytes(priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pub_path.write_bytes(priv.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv_path, pub_path

def create_session(public_key_path):
    pub = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    key = os.urandom(32)
    wrapped = pub.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return key, wrapped

def recover_session(private_key_path, wrapped):
    priv = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    return priv.decrypt(wrapped, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
