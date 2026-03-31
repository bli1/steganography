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
ALPHA = 20.0        # 嵌入强度
BLUR_RADIUS = 0.5   # 模糊攻击半径

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


def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def bits_to_str(bits: np.ndarray) -> str:
    """把 0/1 bit 数组转换成字符串，便于写入 txt。"""
    return "".join(str(int(x)) for x in bits)


# ========= RSA密钥 =========
def ensure_keys():
    """
    如果当前目录下没有 private_key.pem / public_key.pem，就自动生成 1024-bit RSA 密钥对。
    如果你已经有自己的密钥文件，也可以把这个函数在 main 里注释掉。
    """
    if os.path.exists(PRIV_KEY_PATH) and os.path.exists(PUB_KEY_PATH):
        return

    print("Generating RSA keys...")
    pk = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    pub = pk.public_key()

    with open(PRIV_KEY_PATH, "wb") as f:
        f.write(
            pk.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open(PUB_KEY_PATH, "wb") as f:
        f.write(
            pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    print("RSA key pair generated.")


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


# ========= 生成 32-bit 非对称水印 =========
def get_watermark_bits() -> np.ndarray:
    """
    使用 RSA 私钥对 WATERMARK_TEXT 做签名，
    然后对签名做 SHA256，取前 4 字节 = 32 bit 作为水印比特。
    """
    sk = load_private_key()
    sig = sk.sign(
        WATERMARK_TEXT.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    digest = hashlib.sha256(sig).digest()
    wm_bytes = digest[:4]          # 4 字节 → 32 bit
    bits = bytes_to_bits(wm_bytes)  # (32,)
    return bits


# ========= 扩频嵌入 =========
def encode_spread_asym(img_bgr):
    """
    DWT-LL + 扩频 + 32 bit 非对称水印嵌入。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    h, w = LL.shape
    flat = LL.reshape(-1)
    N = flat.size

    wm_bits = get_watermark_bits()
    num_bits = wm_bits.size  # 32

    chunk = N // num_bits    # 每 bit 使用一个很大的块
    if chunk < 20:
        raise ValueError("Image too small for this scheme")

    seed_base = int.from_bytes(hashlib.sha256(b"KEY").digest()[:4], "big")

    for i in range(num_bits):
        start = i * chunk
        end = min(start + chunk, N)
        seg = flat[start:end]

        rng = np.random.RandomState(seed_base + i)
        pn = rng.choice([-1, 1], size=seg.size).astype(np.float32)

        sign = 1.0 if wm_bits[i] == 1 else -1.0
        seg += ALPHA * sign * pn

        flat[start:end] = seg

    LL2 = flat.reshape(h, w)
    rec = pywt.idwt2((LL2, (LH, HL, HH)), "haar")
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR)


# ========= 扩频检测 =========
def detect_spread_asym(img_bgr):
    """
    从图像中提取 32 bit 扩频水印，并与理论水印做汉明距离比较。

    返回:
        verified: bool        是否检测到“合法水印”
        ber: float            bit error rate
        recv_bits: ndarray    解码得到的 bits (32,)
        ref_bits: ndarray     理论正确的 bits (32,)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    h, w = LL.shape
    flat = LL.reshape(-1)
    N = flat.size

    ref_bits = get_watermark_bits()
    num_bits = ref_bits.size  # 32

    chunk = N // num_bits
    seed_base = int.from_bytes(hashlib.sha256(b"KEY").digest()[:4], "big")

    recv_bits = np.zeros(num_bits, dtype=np.uint8)

    for i in range(num_bits):
        start = i * chunk
        end = min(start + chunk, N)
        seg = flat[start:end]

        rng = np.random.RandomState(seed_base + i)
        pn = rng.choice([-1, 1], size=seg.size).astype(np.float32)

        corr = np.sum(seg * pn)
        recv_bits[i] = 1 if corr >= 0 else 0

    errors = np.sum(np.logical_xor(recv_bits, ref_bits))
    ber = errors / num_bits

    verified = ber < 0.25
    return verified, ber, recv_bits, ref_bits


# ========= 单张图像流程 =========
def process_one_image(path):
    img_name = os.path.basename(path)
    print("\n=== Processing:", img_name, "===")

    img = read_cv_image_3ch(path)
    if img is None:
        print("Cannot read image.")
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
    print("No blur:", v1, "BER=%.3f" % ber1)

    with open(os.path.join(out_dec, "no_blur.txt"), "w", encoding="utf-8") as f:
        f.write(f"Verified: {v1}\n")
        f.write(f"BER: {ber1}\n")
        f.write("Decoded bits:\n")
        f.write(bits_to_str(bits_clean) + "\n")
        f.write("Reference bits:\n")
        f.write(bits_to_str(bits_ref) + "\n")

    # ===== 模糊攻击 =====
    blur_img = blur_cv2_bgr_with_pillow(wm_img, BLUR_RADIUS)
    cv2.imwrite(os.path.join(out_enc, "spread_blur.png"), blur_img)

    v2, ber2, bits_blur, _ = detect_spread_asym(blur_img)
    print("After blur:", v2, "BER=%.3f" % ber2)

    with open(os.path.join(out_dec, "blur.txt"), "w", encoding="utf-8") as f:
        f.write(f"Verified: {v2}\n")
        f.write(f"BER: {ber2}\n")
        f.write("Decoded bits:\n")
        f.write(bits_to_str(bits_blur) + "\n")
        f.write("Reference bits:\n")
        f.write(bits_to_str(bits_ref) + "\n")

    # ===== 写一个简单的 Excel 对比表 =====
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Spread_Asym")

    headers = ["Image", "No blur verified", "No blur BER",
               "Blur verified", "Blur BER"]
    for col, h in enumerate(headers):
        ws.write(0, col, h)

    ws.write(1, 0, img_name)
    ws.write(1, 1, str(v1))
    ws.write(1, 2, float(ber1))
    ws.write(1, 3, str(v2))
    ws.write(1, 4, float(ber2))

    wb.save(os.path.join(out_cmp, "Comparison_spread_asym.xls"))

    return v1, v2


# ========= 主函数 =========
def main():
    print(f"WATERMARK_TEXT = {WATERMARK_TEXT}")
    print(f"ALPHA = {ALPHA}  BLUR_RADIUS = {BLUR_RADIUS}")

    ensure_keys()

    if not os.path.isdir(ORIGINAL_DIR):
        print("Original image folder not found:", ORIGINAL_DIR)
        return

    files = [
        f for f in os.listdir(ORIGINAL_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    files.sort()

    if not files:
        print("No images in Original_image.")
        return

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
