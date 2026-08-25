import os
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRIVATE_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUBLIC_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")

private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

with open(PRIVATE_KEY_PATH, "wb") as f:
    f.write(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

public_key = private_key.public_key()

with open(PUBLIC_KEY_PATH, "wb") as f:
    f.write(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

print("RSA 密钥生成完成")
print("Private key size:", private_key.key_size)
print("Public key size:", public_key.key_size)
print("私钥位置:", PRIVATE_KEY_PATH)
print("公钥位置:", PUBLIC_KEY_PATH)