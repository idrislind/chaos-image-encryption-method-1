from chaos_image_encryption.crypto import rsa_backend, ecc_backend

def test_rsa(tmp_path):
    priv,pub=rsa_backend.generate_keypair(tmp_path/'rsa'); s,w=rsa_backend.create_session(pub); assert s==rsa_backend.recover_session(priv,w)

def test_ecc(tmp_path):
    priv,pub=ecc_backend.generate_keypair(tmp_path/'ecc'); s,eph,salt=ecc_backend.create_session(pub); assert s==ecc_backend.recover_session(priv,eph,salt)
