import os
import cv2
import numpy as np
import pywt
import hashlib
from openpyxl import Workbook

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
ALPHA = 20.0        # 嵌入强度
SCALE = 0.8       # 缩放攻击比例（<1缩小，>1放大）

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


def scale_attack(img_bgr: np.ndarray) -> np.ndarray:
    """
    只做缩放几何攻击：先缩放，再拉回原尺寸（保持尺寸一致）
    """
    h, w = img_bgr.shape[:2]
    new_w = max(1, int(w * SCALE))
    new_h = max(1, int(h * SCALE))
    scaled = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    attacked = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
    return attacked


def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def bits_to_str(bits: np.ndarray) -> str:
    return "".join(str(int(x)) for x in bits)


# ========= RSA密钥 =========
def ensure_keys():
    if os.path.exists(PRIV_KEY_PATH) and os.path.exists(PUB_KEY_PATH):
        return
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    with open(PRIV_KEY_PATH, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(PUB_KEY_PATH, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public_key():
    with open(PUB_KEY_PATH, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def sign_message(msg: str) -> bytes:
    priv = load_private_key()
    return priv.sign(msg.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())


def verify_signature(msg: str, sig: bytes) -> bool:
    pub = load_public_key()
    try:
        pub.verify(sig, msg.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
        return True
    except Exception:
        return False


# ========= 扩频水印（DWT域 / Spread Spectrum） =========
def encode_spread_asym(img_bgr):
    sig = sign_message(WATERMARK_TEXT)
    wm_bits = bytes_to_bits(sig).astype(np.int8)
    num_bits = wm_bits.size

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    flat = LH.flatten()
    N = flat.size

    seed_base = int(hashlib.sha256(WATERMARK_TEXT.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)
    chunk = max(1, N // num_bits)

    out = flat.copy()
    for i in range(num_bits):
        start = i * chunk
        end = N if i == num_bits - 1 else min(N, (i + 1) * chunk)
        seg = out[start:end]
        if seg.size == 0:
            break

        rng = np.random.RandomState(seed_base + i)
        pn = rng.choice([-1, 1], size=seg.size).astype(np.float32)

        bit = 1 if wm_bits[i] == 1 else -1
        seg = seg + (ALPHA * bit) * pn
        out[start:end] = seg

    LH2 = out.reshape(LH.shape)
    watermarked = pywt.idwt2((LL, (LH2, HL, HH)), "haar")
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)

    out_bgr = img_bgr.copy()
    out_bgr[:, :, 0] = watermarked
    out_bgr[:, :, 1] = watermarked
    out_bgr[:, :, 2] = watermarked
    return out_bgr


def detect_spread_asym(img_bgr):
    sig = sign_message(WATERMARK_TEXT)
    ref_bits = bytes_to_bits(sig).astype(np.int8)
    num_bits = ref_bits.size

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    flat = LH.flatten()
    N = flat.size

    seed_base = int(hashlib.sha256(WATERMARK_TEXT.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)
    chunk = max(1, N // num_bits)

    recv_bits = np.zeros(num_bits, dtype=np.int8)

    for i in range(num_bits):
        start = i * chunk
        end = N if i == num_bits - 1 else min(N, (i + 1) * chunk)
        seg = flat[start:end]
        if seg.size == 0:
            break

        rng = np.random.RandomState(seed_base + i)
        pn = rng.choice([-1, 1], size=seg.size).astype(np.float32)

        corr = float(np.sum(seg * pn))
        recv_bits[i] = 1 if corr >= 0 else 0

    errors = int(np.sum(np.logical_xor(recv_bits, ref_bits)))
    ber = errors / num_bits
    verified = ber < 0.25
    return verified, ber, recv_bits, ref_bits


# ========= 单张图片流程 =========
def process_one_image(path):
    img_name = os.path.basename(path)
    print("\n=== Spread Processing:", img_name, "===")

    img = read_cv_image_3ch(path)
    if img is None:
        print("  [Error] cannot read image.")
        return False, False

    name_no_ext, _ = os.path.splitext(img_name)

    out_enc = os.path.join(SPREAD_ENC_DIR, name_no_ext)
    out_dec = os.path.join(SPREAD_DEC_DIR, name_no_ext)
    out_cmp = os.path.join(SPREAD_COMP_DIR, name_no_ext)
    os.makedirs(out_enc, exist_ok=True)
    os.makedirs(out_dec, exist_ok=True)
    os.makedirs(out_cmp, exist_ok=True)

    # ===== 嵌入 =====
    wm_img = encode_spread_asym(img)
    cv2.imwrite(os.path.join(out_enc, "spread.png"), wm_img)

    # ===== 检测（无攻击）=====
    v1, ber1, bits_clean, bits_ref = detect_spread_asym(wm_img)
    print("No attack:", v1, "BER=%.3f" % ber1)

    with open(os.path.join(out_dec, "no_attack.txt"), "w", encoding="utf-8") as f:
        f.write(f"Verified: {v1}\n")
        f.write(f"BER: {ber1}\n")
        f.write("Decoded bits:\n")
        f.write(bits_to_str(bits_clean) + "\n")
        f.write("Reference bits:\n")
        f.write(bits_to_str(bits_ref) + "\n")

    # ===== 缩放几何攻击（Scale） =====
    geo_img = scale_attack(wm_img)
    cv2.imwrite(os.path.join(out_enc, "spread_geo.png"), geo_img)

    v2, ber2, bits_geo, _ = detect_spread_asym(geo_img)
    print("After geo(scale):", v2, "BER=%.3f" % ber2)

    with open(os.path.join(out_dec, "geo_scale.txt"), "w", encoding="utf-8") as f:
        f.write(f"Verified: {v2}\n")
        f.write(f"BER: {ber2}\n")
        f.write("Decoded bits:\n")
        f.write(bits_to_str(bits_geo) + "\n")
        f.write("Reference bits:\n")
        f.write(bits_to_str(bits_ref) + "\n")

    # ===== 写一个简单的 Excel 对比表 =====
    wb = Workbook()
    ws = wb.active
    ws.title = "Spread_Asym"

    headers = ["Image", "No attack verified", "No attack BER",
               "Geo(scale) verified", "Geo(scale) BER"]
    ws.append(headers)
    ws.append([img_name, str(v1), float(ber1), str(v2), float(ber2)])

    wb.save(os.path.join(out_cmp, "Comparison_spread_asym.xlsx"))

    return v1, v2


# ========= 主函数 =========
def main():
    print("===== DWT-BASED RSA SPREAD-SPECTRUM WATERMARKING =====")
    print(f"WATERMARK_TEXT = {WATERMARK_TEXT}")
    print(f"ALPHA = {ALPHA}  SCALE = {SCALE}")

    ensure_keys()

    if not os.path.isdir(ORIGINAL_DIR):
        print("Original image folder not found:", ORIGINAL_DIR)
        return

    files = [
        f for f in os.listdir(ORIGINAL_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    files.sort()

    if not files:
        print("No images in Original_image.")
        return

    fail1 = []
    fail2 = []

    for fname in files:
        v1, v2 = process_one_image(os.path.join(ORIGINAL_DIR, fname))
        if not v1:
            fail1.append(fname)
        if not v2:
            fail2.append(fname)

    print("\n============== Spread Summary ==============")
    print("No attack fail:", len(fail1), fail1)
    print("Geo(scale) fail:", len(fail2), fail2)


if __name__ == "__main__":
    main()