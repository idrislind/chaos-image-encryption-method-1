import os, numpy as np
from chaos_image_encryption.cipher.core import encrypt_array, decrypt_array

def test_gray_roundtrip():
    r=np.random.default_rng(1); img=r.integers(0,256,(20,27),dtype=np.uint8); k=bytes(range(32)); n=bytes(range(16))
    c=encrypt_array(img,k,n); d=decrypt_array(c,k,n)
    assert not np.array_equal(img,c); assert np.array_equal(img,d)

def test_rgb_roundtrip():
    r=np.random.default_rng(2); img=r.integers(0,256,(17,21,3),dtype=np.uint8); k=os.urandom(32); n=os.urandom(16)
    assert np.array_equal(img, decrypt_array(encrypt_array(img,k,n),k,n))
