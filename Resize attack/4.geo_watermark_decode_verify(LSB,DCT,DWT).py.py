import os
from openpyxl import Workbook
import cv2
import numpy as np
import binascii
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
import pywt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 已经做完缩放攻击的图片
GEO_DIR = os.path.join(BASE_DIR, "Encoded_geo")

# decode 输出
DECODED_GEO_BASE_DIR = os.path.join(BASE_DIR, "Decoded_geo_output")

# 统计结果
COMPARISON_GEO_BASE_DIR = os.path.join(BASE_DIR, "Comparison_geo_result")

# 密钥保持一致
PRIV_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUB_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")
WATERMARK_TEXT = "ASys Encryption"


# ===== RSA 工具 =====
def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        data = f.read()
    return serialization.load_pem_private_key(data, password=None)


def load_public_key():
    with open(PUB_KEY_PATH, "rb") as f:
        data = f.read()
    return serialization.load_pem_public_key(data)


def sign_message(msg: str) -> bytes:
    private_key = load_private_key()
    return private_key.sign(
        msg.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def verify_signature(msg: str, sig_bytes: bytes) -> bool:
    try:
        public_key = load_public_key()
        public_key.verify(
            sig_bytes,
            msg.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


# 辅助用的函数
def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits[: len(bits) // 8 * 8]
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def read_cv(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# ====== 三种 decode 方法,跟blur一样 ======
def decode_lsb(img) -> bytes:
    flat = img.reshape(-1, 3)
    header_bits = "".join(str(flat[i, 0] & 1) for i in range(32))
    length = int(header_bits, 2)

    capacity = flat.shape[0] - 32
    if length <= 0 or length > capacity:
        return b""

    bits = "".join(str(flat[i, 0] & 1) for i in range(32, 32 + length))
    return bits_to_bytes(bits)


def decode_dct(img, bits_len: int, Q=8.0, bits_per_block=4) -> bytes:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blocks_y, blocks_x = h // 8, w // 8
    bits = ""
    bit_idx = 0
    pos = [(4, 4), (3, 3), (2, 2), (5, 5)]

    for by in range(blocks_y):
        for bx in range(blocks_x):
            if bit_idx >= bits_len:
                break
            y0, x0 = by * 8, bx * 8
            block = np.float32(gray[y0:y0 + 8, x0:x0 + 8])
            dct_block = cv2.dct(block)
            for ry, rx in pos:
                if bit_idx >= bits_len:
                    break
                q = int(np.round(dct_block[ry, rx] / Q))
                bits += str(q & 1)
                bit_idx += 1
        if bit_idx >= bits_len:
            break

    return bits_to_bytes(bits)


def decode_dwt(img, bits_len: int, Q=8.0) -> bytes:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LL, (LH, HL, HH) = pywt.dwt2(np.float32(gray), "haar")
    flat = LH.flatten()

    bits = ""
    length = min(bits_len, flat.size)
    for i in range(length):
        q = int(np.round(flat[i] / Q))
        bits += str(q & 1)

    return bits_to_bytes(bits)


def save_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(content))


# ====== 对单个几何攻击文件夹 Decode ======
def process_geo_folder(folder_name, sig_bits_len):

    print("====================================")
    print(f"Processing geo folder: {folder_name}")

    folder_path = os.path.join(GEO_DIR, folder_name)

    dec_out = os.path.join(DECODED_GEO_BASE_DIR, folder_name)
    os.makedirs(dec_out, exist_ok=True)

    comp_out = os.path.join(COMPARISON_GEO_BASE_DIR, folder_name)
    os.makedirs(comp_out, exist_ok=True)

    # 结果
    lsb_valid = dct_valid = dwt_valid = False

    # ---------- LSB ----------
    lsb_path = os.path.join(folder_path, "LSB_geo.png")
    if os.path.exists(lsb_path):
        img = read_cv(lsb_path)
        dec_bytes = decode_lsb(img)
        save_text(os.path.join(dec_out, "LSB_geo_decoded.txt"),
                  binascii.hexlify(dec_bytes).decode())
        lsb_valid = verify_signature(WATERMARK_TEXT, dec_bytes)
        print("  LSB valid after geo:", lsb_valid)

    # ---------- DCT ----------
    dct_path = os.path.join(folder_path, "DCT_geo.png")
    if os.path.exists(dct_path):
        img = read_cv(dct_path)
        dec_bytes = decode_dct(img, sig_bits_len)
        save_text(os.path.join(dec_out, "DCT_geo_decoded.txt"),
                  binascii.hexlify(dec_bytes).decode())
        dct_valid = verify_signature(WATERMARK_TEXT, dec_bytes)
        print("  DCT valid after geo:", dct_valid)

    # ---------- DWT ----------
    dwt_path = os.path.join(folder_path, "DWT_geo.png")
    if os.path.exists(dwt_path):
        img = read_cv(dwt_path)
        dec_bytes = decode_dwt(img, sig_bits_len)
        save_text(os.path.join(dec_out, "DWT_geo_decoded.txt"),
                  binascii.hexlify(dec_bytes).decode())
        dwt_valid = verify_signature(WATERMARK_TEXT, dec_bytes)
        print("  DWT valid after geo:", dwt_valid)

    # 结果写 EXCEL
    wb = Workbook()
    ws = wb.active
    ws.title = "Geo_Comparison"

    headers = ["Folder", "LSB valid", "DCT valid", "DWT valid"]
    ws.append(headers)
    ws.append([folder_name, str(lsb_valid), str(dct_valid), str(dwt_valid)])

    wb.save(os.path.join(comp_out, "comparison_geo.xlsx"))

    return lsb_valid, dct_valid, dwt_valid


def main():
    # 得到签名长度（与 encode 时一致）
    sig_bytes = sign_message(WATERMARK_TEXT)
    sig_bits_len = len(bytes_to_bits(sig_bytes))
    print("Signature bits:", sig_bits_len)

    if not os.path.isdir(GEO_DIR):
        print("Geo folder not found:", GEO_DIR)
        return

    folders = sorted(
        f for f in os.listdir(GEO_DIR)
        if os.path.isdir(os.path.join(GEO_DIR, f))
    )

    if not folders:
        print("No geo folders found.")
        return

    lsb_false, dct_false, dwt_false = [], [], []

    for f in folders:
        lsb_ok, dct_ok, dwt_ok = process_geo_folder(f, sig_bits_len)
        if not lsb_ok:
            lsb_false.append(f)
        if not dct_ok:
            dct_false.append(f)
        if not dwt_ok:
            dwt_false.append(f)

    print("====================================")
    print("After geo decode false count:")
    print("LSB false:", len(lsb_false), lsb_false)
    print("DCT false:", len(dct_false), dct_false)
    print("DWT false:", len(dwt_false), dwt_false)
    print("Done.")


if __name__ == "__main__":
    main()