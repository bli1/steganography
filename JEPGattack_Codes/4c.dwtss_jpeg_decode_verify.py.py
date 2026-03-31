import os
import csv
import xlwt
import cv2
import numpy as np
import hashlib
import pywt
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JPEG_DIR = os.path.join(BASE_DIR, "Spread_jpeg")
COMPARISON_DIR = os.path.join(BASE_DIR, "Comparison_dwtss_jpeg_result")
PRIV_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUB_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")
WATERMARK_TEXT = "ASys Encryption"
ALPHA = 20.0

def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def sign_message(msg: str) -> bytes:
    sk = load_private_key()
    return sk.sign(msg.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())

def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

def get_watermark_bits() -> np.ndarray:
    sig = sign_message(WATERMARK_TEXT)
    digest = hashlib.sha256(sig).digest()
    return bytes_to_bits(digest[:4])

def detect_spread_asym(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    flat = LL.reshape(-1)
    N = flat.size
    ref_bits = get_watermark_bits()
    num_bits = ref_bits.size
    if num_bits == 0:
        return False, 1.0
    chunk = N // num_bits
    if chunk <= 0:
        return False, 1.0
    seed_base = int.from_bytes(hashlib.sha256(b"KEY").digest()[:4], "big")
    recv = np.zeros(num_bits, dtype=np.uint8)
    for i in range(num_bits):
        start = i * chunk
        end = min(start + chunk, N)
        seg = flat[start:end]
        if seg.size == 0:
            recv[i] = 0
            continue
        rng = np.random.RandomState(seed_base + i)
        pn = rng.choice([-1, 1], size=seg.size).astype(np.float32)
        corr = float(np.sum(seg * pn))
        recv[i] = 1 if corr >= 0 else 0
    errors = int(np.sum(np.logical_xor(recv, ref_bits)))
    ber = errors / num_bits
    return (ber < 0.25), float(ber)

def list_quality_dirs():
    if not os.path.isdir(JPEG_DIR):
        return []
    out = []
    for d in os.listdir(JPEG_DIR):
        p = os.path.join(JPEG_DIR, d)
        if os.path.isdir(p) and d.startswith("Q"):
            try:
                out.append((int(d[1:]), p))
            except Exception:
                pass
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def main():
    if not os.path.exists(PRIV_KEY_PATH) or not os.path.exists(PUB_KEY_PATH):
        print("Key files not found. Run your embed script first to generate private_key.pem and public_key.pem.")
        return
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    q_dirs = list_quality_dirs()
    if not q_dirs:
        print("No JPEG folders found in:", JPEG_DIR)
        return
    per_image_csv = os.path.join(COMPARISON_DIR, "per_image_dwtss_jpeg.csv")
    summary_csv = os.path.join(COMPARISON_DIR, "summary_dwtss_jpeg.csv")
    summary_xls = os.path.join(COMPARISON_DIR, "summary_dwtss_jpeg.xls")
    per_image_rows = []
    summary_rows = []
    for q, q_path in q_dirs:
        folders = sorted([f for f in os.listdir(q_path) if os.path.isdir(os.path.join(q_path, f))])
        total = 0
        ok = 0
        ber_sum = 0.0
        for folder in folders:
            img_path = os.path.join(q_path, folder, "spread.jpg")
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            total += 1
            verified, ber = detect_spread_asym(img)
            if verified:
                ok += 1
            ber_sum += float(ber)
            per_image_rows.append([q, folder, bool(verified), float(ber)])
        ok_rate = (ok / total) if total else 0.0
        mean_ber = (ber_sum / total) if total else 0.0
        summary_rows.append([q, total, ok, ok_rate, mean_ber])
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jpeg_quality", "image_folder", "verified", "ber"])
        w.writerows(per_image_rows)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jpeg_quality", "total_images", "verified_count", "verified_rate", "mean_ber"])
        w.writerows(summary_rows)
    wb = xlwt.Workbook()
    ws = wb.add_sheet("summary")
    headers = ["jpeg_quality", "total_images", "verified_count", "verified_rate", "mean_ber"]
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
