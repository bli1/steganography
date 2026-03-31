import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import pywt
import hashlib
import xlwt

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


# ========= 路径 =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(BASE_DIR, "Original_image")

SPREAD_ENC_DIR = os.path.join(BASE_DIR, "Spread_encoded")
SPREAD_DEC_DIR = os.path.join(BASE_DIR, "Spread_decoded")
SPREAD_COMP_DIR = os.path.join(BASE_DIR, "Spread_comparison")

os.makedirs(SPREAD_ENC_DIR, exist_ok=True)
os.makedirs(SPREAD_DEC_DIR, exist_ok=True)
os.makedirs(SPREAD_COMP_DIR, exist_ok=True)

# ========= 参数 =========
WATERMARK_TEXT = "ASys Encryption"
ALPHA = 20.0        # 🔥 提高强度
BLUR_RADIUS = 0.5

PRIV_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUB_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")


# ========= 工具 =========
def read_cv_image_3ch(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def blur_cv2_bgr_with_pillow(img, r):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil2 = pil.filter(ImageFilter.GaussianBlur(radius=r))
    return cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)


def bytes_to_bits(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


# ========= RSA密钥 =========
def ensure_keys():
    if os.path.exists(PRIV_KEY_PATH) and os.path.exists(PUB_KEY_PATH):
        return
    print("Generating RSA keys...")
    pk = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    pub = pk.public_key()

    with open(PRIV_KEY_PATH, "wb") as f:
        f.write(pk.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open(PUB_KEY_PATH, "wb") as f:
        f.write(pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


# ========= 生成32-bit 非对称水印 =========
def get_watermark_bits():
    """
    使用 RSA 私钥对 WATERMARK_TEXT 做签名，
    再 SHA256 → 取前 4 字节 = 32 bit
    """
    sk = load_private_key()
    sig = sk.sign(
        WATERMARK_TEXT.encode(),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    digest = hashlib.sha256(sig).digest()
    wm_bytes = digest[:4]          # 4 字节 → 32 bit
    bits = bytes_to_bits(wm_bytes)
    return bits  # np.array(..., dtype=uint8), length = 32


# ========= 扩频嵌入 =========
def encode_spread_asym(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    h, w = LL.shape
    flat = LL.reshape(-1)
    N = flat.size

    wm_bits = get_watermark_bits()
    num_bits = wm_bits.size   # 32 bit

    chunk = N // num_bits     # 每 bit 使用一个非常大的块
    if chunk < 20:
        raise ValueError("Image too small")

    seed_base = int.from_bytes(hashlib.sha256(b"KEY").digest()[:4], "big")

    for i in range(num_bits):
        start = i * chunk
        end = min(start + chunk, N)
        seg = flat[start:end]

        rng = np.random.RandomState(seed_base + i)
        pn = rng.choice([-1, 1], size=seg.size).astype(np.float32)

        sign = 1 if wm_bits[i] == 1 else -1
        seg += ALPHA * sign * pn
        flat[start:end] = seg

    LL2 = flat.reshape(h, w)
    img2 = pywt.idwt2((LL2, (LH, HL, HH)), "haar")
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)


# ========= 扩频检测 =========
def detect_spread_asym(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    h, w = LL.shape
    flat = LL.reshape(-1)
    N = flat.size

    wm_bits = get_watermark_bits()      # 正确 32bit 水印
    num_bits = wm_bits.size

    chunk = N // num_bits
    seed_base = int.from_bytes(hashlib.sha256(b"KEY").digest()[:4], "big")

    recv = np.zeros(num_bits, dtype=np.uint8)

    for i in range(num_bits):
        start = i * chunk
        end = min(start + chunk, N)
        seg = flat[start:end]

        rng = np.random.RandomState(seed_base + i)
        pn = rng.choice([-1, 1], size=seg.size).astype(np.float32)

        corr = np.sum(seg * pn)
        recv[i] = 1 if corr >= 0 else 0

    # === BER ===
    errors = np.sum(np.logical_xor(recv, wm_bits))
    ber = errors / num_bits

    # 阈值：小于 0.25 就算 True
    verified = ber < 0.25
    return verified, ber


# ========= 单张图像流程 =========
def process_one_image(path):
    img_name = os.path.basename(path)
    print("\n=== Processing:", img_name, "===")

    img = read_cv_image_3ch(path)
    if img is None:
        print("Cannot read image.")
        return False, False

    # 输出路径
    name_no_ext, _ = os.path.splitext(img_name)
    out_enc = os.path.join(SPREAD_ENC_DIR, name_no_ext)
    out_dec = os.path.join(SPREAD_DEC_DIR, name_no_ext)
    os.makedirs(out_enc, exist_ok=True)
    os.makedirs(out_dec, exist_ok=True)

    # ===== 嵌入 =====
    wm_img = encode_spread_asym(img)
    cv2.imwrite(os.path.join(out_enc, "spread.png"), wm_img)

    # ===== 检测（无攻击）=====
    v1, ber1 = detect_spread_asym(wm_img)
    print("No blur:", v1, "BER=%.3f" % ber1)

    # ===== 模糊攻击 =====
    blur_img = blur_cv2_bgr_with_pillow(wm_img, BLUR_RADIUS)
    cv2.imwrite(os.path.join(out_enc, "spread_blur.png"), blur_img)

    v2, ber2 = detect_spread_asym(blur_img)
    print("After blur:", v2, "BER=%.3f" % ber2)

    return v1, v2


# ========= 主函数 =========
def main():
    ensure_keys()

    files = [
        f for f in os.listdir(ORIGINAL_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    files.sort()

    fail1 = []
    fail2 = []

    for f in files:
        v1, v2 = process_one_image(os.path.join(ORIGINAL_DIR, f))
        if not v1:
            fail1.append(f)
        if not v2:
            fail2.append(f)

    print("\n============== Summary ==============")
    print("No attack fail:", len(fail1), fail1)
    print("Blur fail:", len(fail2), fail2)


if __name__ == "__main__":
    main()
