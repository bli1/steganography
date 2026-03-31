from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# 1. 生成 RSA 私钥（2048 位够用）
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=1024,
)

# 2. 导出并保存私钥（PEM）
with open("private_key.pem", "wb") as f:
    f.write(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),  # 不加密码，简单点
        )
    )

# 3. 导出并保存公钥（PEM）
public_key = private_key.public_key()
with open("public_key.pem", "wb") as f:
    f.write(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

print("生成完成：private_key.pem 和 public_key.pem")
