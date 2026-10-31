import numpy as np
from chaos_image_encryption.chaos.henon import generate_henon
from chaos_image_encryption.chaos.permutation import permute_pixels, inverse_permute_pixels
from chaos_image_encryption.chaos.diffusion import diffuse, inverse_diffuse, diffuse_v3, inverse_diffuse_v3

def test_henon_reproducible():
    a,b=generate_henon(.01,.02,64); c,d=generate_henon(.01,.02,64); assert np.array_equal(a,c) and np.array_equal(b,d)

def test_permutation_roundtrip():
    img=np.arange(60,dtype=np.uint8).reshape(4,5,3); p=np.array([19,0,5,1,6,2,7,3,8,4,9,10,11,12,13,14,15,16,17,18]); assert np.array_equal(img,inverse_permute_pixels(permute_pixels(img,p),p))

def test_diffusion_roundtrip():
    r=np.random.default_rng(3); p=r.integers(0,256,2000,dtype=np.uint8); k1=r.integers(0,256,2000,dtype=np.uint8); k2=r.integers(0,256,2000,dtype=np.uint8)
    assert np.array_equal(p,inverse_diffuse(diffuse(p,k1,k2,13,211),k1,k2,13,211))

from chaos_image_encryption.chaos.diffusion import diffuse_v1, inverse_diffuse_v1


def test_legacy_diffusion_roundtrip():
    r=np.random.default_rng(31); p=r.integers(0,256,4096,dtype=np.uint8); k1=r.integers(0,256,4096,dtype=np.uint8); k2=r.integers(0,256,4096,dtype=np.uint8)
    assert np.array_equal(p,inverse_diffuse_v1(diffuse_v1(p,k1,k2,17,203),k1,k2,17,203))


def test_nonlinear_diffusion_has_broad_one_bit_avalanche():
    r=np.random.default_rng(99)
    p=r.integers(0,256,8192,dtype=np.uint8)
    q=p.copy(); q[len(q)//2] ^= np.uint8(1)
    k1=r.integers(0,256,len(p),dtype=np.uint8); k2=r.integers(0,256,len(p),dtype=np.uint8)
    a=diffuse(p,k1,k2,33,219); b=diffuse(q,k1,k2,33,219)
    changed=np.mean(a != b) * 100.0
    intensity=np.mean(np.abs(a.astype(np.int16)-b.astype(np.int16))) / 255.0 * 100.0
    assert changed > 98.0
    assert 25.0 < intensity < 42.0


def test_v3_diffusion_roundtrip():
    r=np.random.default_rng(303)
    p=r.integers(0,256,5000,dtype=np.uint8)
    k1=r.integers(0,256,5000,dtype=np.uint8)
    k2=r.integers(0,256,5000,dtype=np.uint8)
    c=diffuse_v3(p,k1,k2,61,197)
    assert np.array_equal(p,inverse_diffuse_v3(c,k1,k2,61,197))


def test_v3_differential_response_stable_on_fixed_trials():
    r=np.random.default_rng(404)
    p=r.integers(0,256,12288,dtype=np.uint8)
    q=p.copy(); q[len(q)//2] ^= np.uint8(1)
    values=[]
    for trial in range(12):
        rr=np.random.default_rng(1000+trial)
        k1=rr.integers(0,256,len(p),dtype=np.uint8)
        k2=rr.integers(0,256,len(p),dtype=np.uint8)
        a=diffuse_v3(p,k1,k2,33+2*trial,219-2*trial)
        b=diffuse_v3(q,k1,k2,33+2*trial,219-2*trial)
        changed=np.mean(a != b) * 100.0
        intensity=np.mean(np.abs(a.astype(np.int16)-b.astype(np.int16))) / 255.0 * 100.0
        values.append((changed,intensity))
    assert min(v[0] for v in values) > 99.0
    assert min(v[1] for v in values) > 30.0
    assert max(v[1] for v in values) < 37.0
