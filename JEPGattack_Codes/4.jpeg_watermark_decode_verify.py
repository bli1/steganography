import os
import csv
import xlwt
import cv2
import numpy as np
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
import pywt

def pick_base_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if os.path.exists(os.path.join(here, "private_key.pem")) or os.path.isdir(os.path.join(here, "Encoded_jpeg")):
        return here
    if os.path.exists(os.path.join(parent, "private_key.pem")) or os.path.isdir(os.path.join(parent, "Encoded_jpeg")):
        return parent
    return parent

BASE_DIR = pick_base_dir()

JPEG_DIR = os.path.join(BASE_DIR, "Encoded_jpeg")
DECODED_JPEG_BASE_DIR = os.path.join(BASE_DIR, "Decoded_jpeg_output")
COMPARISON_JPEG_BASE_DIR = os.path.join(BASE_DIR, "Comparison_jpeg_result")

PRIV_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUB_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")
WATERMARK_TEXT = "ASys Encryption"

def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def load_public_key():
    with open(PUB_KEY_PATH, "rb") as f:
        return serialization.load_pem_public_key(f.read())

def sign_message(msg: str) -> bytes:
    private_key = load_private_key()
    return private_key.sign(msg.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())

def verify_signature(msg: str, sig_bytes: bytes) -> bool:
    try:
        public_key = load_public_key()
        public_key.verify(sig_bytes, msg.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
        return True
    except Exception:
        return False

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

def decode_lsb_safe(img) -> bytes:
    flat = img.reshape(-1, 3)
    if flat.shape[0] < 32:
        return b""
    header_bits = "".join(str(flat[i, 0] & 1) for i in range(32))
    try:
        length = int(header_bits, 2)
    except Exception:
        return b""
    max_len = flat.shape[0] - 32
    if length <= 0 or length > max_len:
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
            for ry, rx in pos[:bits_per_block]:
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
    L = min(bits_len, flat.size)
    bits = ""
    for i in range(L):
        q = int(np.round(flat[i] / Q))
        bits += str(q & 1)
    return bits_to_bytes(bits)

def list_quality_dirs():
    if not os.path.isdir(JPEG_DIR):
        return []
    out = []
    for d in os.listdir(JPEG_DIR):
        p = os.path.join(JPEG_DIR, d)
        if os.path.isdir(p) and d.startswith("Q"):
            try:
                q = int(d[1:])
                out.append((q, p))
            except Exception:
                pass
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def main():
    os.makedirs(DECODED_JPEG_BASE_DIR, exist_ok=True)
    os.makedirs(COMPARISON_JPEG_BASE_DIR, exist_ok=True)

    if not os.path.exists(PRIV_KEY_PATH) or not os.path.exists(PUB_KEY_PATH):
        print("Key files not found:", PRIV_KEY_PATH, PUB_KEY_PATH)
        return

    sig_bits_len = len(bytes_to_bits(sign_message(WATERMARK_TEXT)))

    q_dirs = list_quality_dirs()
    if not q_dirs:
        print("No JPEG folders found in:", JPEG_DIR)
        return

    per_image_csv = os.path.join(COMPARISON_JPEG_BASE_DIR, "per_image_jpeg.csv")
    summary_csv = os.path.join(COMPARISON_JPEG_BASE_DIR, "summary_jpeg.csv")
    summary_xls = os.path.join(COMPARISON_JPEG_BASE_DIR, "summary_jpeg.xls")

    per_image_rows = []
    summary_rows = []

    for q, q_path in q_dirs:
        folders = sorted([
            f for f in os.listdir(q_path)
            if os.path.isdir(os.path.join(q_path, f))
        ])

        total = 0
        lsb_ok = 0
        dct_ok = 0
        dwt_ok = 0

        for folder in folders:
            folder_path = os.path.join(q_path, folder)
            total += 1

            lsb_valid = False
            dct_valid = False
            dwt_valid = False

            lsb_path = os.path.join(folder_path, "LSB.jpg")
            if os.path.exists(lsb_path):
                img = read_cv(lsb_path)
                if img is not None:
                    dec = decode_lsb_safe(img)
                    lsb_valid = verify_signature(WATERMARK_TEXT, dec)

            dct_path = os.path.join(folder_path, "DCT.jpg")
            if os.path.exists(dct_path):
                img = read_cv(dct_path)
                if img is not None:
                    dec = decode_dct(img, sig_bits_len)
                    dct_valid = verify_signature(WATERMARK_TEXT, dec)

            dwt_path = os.path.join(folder_path, "DWT.jpg")
            if os.path.exists(dwt_path):
                img = read_cv(dwt_path)
                if img is not None:
                    dec = decode_dwt(img, sig_bits_len)
                    dwt_valid = verify_signature(WATERMARK_TEXT, dec)

            if lsb_valid:
                lsb_ok += 1
            if dct_valid:
                dct_ok += 1
            if dwt_valid:
                dwt_ok += 1

            per_image_rows.append([q, folder, lsb_valid, dct_valid, dwt_valid])

        lsb_rate = (lsb_ok / total) if total else 0.0
        dct_rate = (dct_ok / total) if total else 0.0
        dwt_rate = (dwt_ok / total) if total else 0.0
        summary_rows.append([q, total, lsb_ok, lsb_rate, dct_ok, dct_rate, dwt_ok, dwt_rate])

    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jpeg_quality", "image_folder", "lsb_valid", "dct_valid", "dwt_valid"])
        w.writerows(per_image_rows)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jpeg_quality", "total_images",
                    "lsb_valid_count", "lsb_valid_rate",
                    "dct_valid_count", "dct_valid_rate",
                    "dwt_valid_count", "dwt_valid_rate"])
        w.writerows(summary_rows)

    wb = xlwt.Workbook()
    ws = wb.add_sheet("summary")
    headers = ["jpeg_quality", "total_images",
               "lsb_valid_count", "lsb_valid_rate",
               "dct_valid_count", "dct_valid_rate",
               "dwt_valid_count", "dwt_valid_rate"]
    for c, h in enumerate(headers):
        ws.write(0, c, h)
    for r, row in enumerate(summary_rows, start=1):
        for c, val in enumerate(row):
            ws.write(r, c, val)
    wb.save(summary_xls)

    print("Saved:", per_image_csv)
    print("Saved:", summary_csv)
    print("Saved:", summary_xls)

if __name__ == "__main__":
    main()
