import os
from openpyxl import Workbook
import cv2
import numpy as np
from PIL import Image
import binascii
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
import pywt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(BASE_DIR, "Original_image")
ENCODED_BASE_DIR = os.path.join(BASE_DIR, "Encoded_image")
COMPARISON_BASE_DIR = os.path.join(BASE_DIR, "Comparison_result")
DECODED_BASE_DIR = os.path.join(BASE_DIR, "Decoded_output")

PRIV_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUB_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")

WATERMARK_TEXT = "ASys Encryption"


def read_cv_image_3ch(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_keys():
    if os.path.exists(PRIV_KEY_PATH) and os.path.exists(PUB_KEY_PATH):
        return
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=1024,
    )
    public_key = private_key.public_key()
    with open(PRIV_KEY_PATH, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(PUB_KEY_PATH, "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        data = f.read()
    return serialization.load_pem_private_key(data, password=None)


def load_public_key():
    with open(PUB_KEY_PATH, "rb") as f:
        data = f.read()
    return serialization.load_pem_public_key(data)


def sign_message(msg: str) -> bytes:
    ensure_keys()
    private_key = load_private_key()
    signature = private_key.sign(
        msg.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return signature


def verify_signature(msg: str, sig_bytes: bytes) -> bool:
    try:
        public_key = load_public_key()
    except Exception:
        return False
    try:
        public_key.verify(
            sig_bytes,
            msg.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits[: len(bits) // 8 * 8]
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def lsb_capacity(img) -> int:
    h, w, _ = img.shape
    return h * w


def encode_lsb(img, data: bytes):
    bits = bytes_to_bits(data)
    length = len(bits)
    total_bits = 32 + length
    cap = lsb_capacity(img)
    if total_bits > cap:
        return False, None
    flat = img.reshape(-1, 3).copy()
    header = f"{length:032b}"
    all_bits = header + bits
    for i, bit in enumerate(all_bits):
        b = flat[i, 0]
        flat[i, 0] = (b & 0xFE) | int(bit)
    watermarked = flat.reshape(img.shape)
    return True, watermarked


def decode_lsb(img) -> bytes:
    flat = img.reshape(-1, 3)
    header_bits = ""
    for i in range(32):
        header_bits += str(flat[i, 0] & 1)
    length = int(header_bits, 2)
    bits = ""
    for i in range(32, 32 + length):
        bits += str(flat[i, 0] & 1)
    return bits_to_bytes(bits)


def encode_dct(img, data: bytes, Q: float = 8.0, bits_per_block: int = 4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blocks_y = h // 8
    blocks_x = w // 8
    bits = bytes_to_bits(data)
    length = len(bits)
    capacity = blocks_x * blocks_y * bits_per_block
    if length > capacity:
        return False, None
    work = np.float32(gray.copy())
    bit_idx = 0
    positions = [(4, 4), (3, 3), (2, 2), (5, 5)]
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if bit_idx >= length:
                break
            y0 = by * 8
            x0 = bx * 8
            block = work[y0:y0 + 8, x0:x0 + 8]
            dct_block = cv2.dct(block)
            for pi in range(bits_per_block):
                if bit_idx >= length:
                    break
                (ry, rx) = positions[pi]
                coef = dct_block[ry, rx]
                bit = int(bits[bit_idx])
                q = np.round(coef / Q)
                if int(q) & 1 != bit:
                    if coef >= 0:
                        q += 1
                    else:
                        q -= 1
                coef_new = q * Q
                dct_block[ry, rx] = coef_new
                bit_idx += 1
            block_rec = cv2.idct(dct_block)
            work[y0:y0 + 8, x0:x0 + 8] = block_rec
        if bit_idx >= length:
            break
    watermarked_gray = np.clip(work, 0, 255).astype("uint8")
    watermarked = cv2.cvtColor(watermarked_gray, cv2.COLOR_GRAY2BGR)
    return True, watermarked


def decode_dct(img, bits_len: int, Q: float = 8.0, bits_per_block: int = 4) -> bytes:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blocks_y = h // 8
    blocks_x = w // 8
    bits = ""
    bit_idx = 0
    positions = [(4, 4), (3, 3), (2, 2), (5, 5)]
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if bit_idx >= bits_len:
                break
            y0 = by * 8
            x0 = bx * 8
            block = np.float32(gray[y0:y0 + 8, x0:x0 + 8])
            dct_block = cv2.dct(block)
            for pi in range(bits_per_block):
                if bit_idx >= bits_len:
                    break
                (ry, rx) = positions[pi]
                coef = dct_block[ry, rx]
                q = int(np.round(coef / Q))
                bit = q & 1
                bits += str(bit)
                bit_idx += 1
        if bit_idx >= bits_len:
            break
    return bits_to_bytes(bits)


def encode_dwt(img, data: bytes, Q: float = 8.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bits = bytes_to_bits(data)
    length = len(bits)
    coeffs2 = pywt.dwt2(np.float32(gray), "haar")
    LL, (LH, HL, HH) = coeffs2
    flat = LH.flatten()
    if length > flat.size:
        return False, None
    flat = flat.copy()
    for i in range(length):
        coef = flat[i]
        bit = int(bits[i])
        q = np.round(coef / Q)
        if int(q) & 1 != bit:
            if coef >= 0:
                q += 1
            else:
                q -= 1
        flat[i] = q * Q
    LH_mark = flat.reshape(LH.shape)
    coeffs_mark = (LL, (LH_mark, HL, HH))
    rec = pywt.idwt2(coeffs_mark, "haar")
    rec = np.clip(rec, 0, 255).astype("uint8")
    watermarked = cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR)
    return True, watermarked


def decode_dwt(img, bits_len: int, Q: float = 8.0) -> bytes:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(np.float32(gray), "haar")
    LL, (LH, HL, HH) = coeffs2
    flat = LH.flatten()
    bits = ""
    L = min(bits_len, flat.size)
    for i in range(L):
        coef = flat[i]
        q = int(np.round(coef / Q))
        bit = q & 1
        bits += str(bit)
    return bits_to_bytes(bits)


def save_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(content))


def process_one_image(img_path):
    img_name = os.path.basename(img_path)
    name_no_ext, _ = os.path.splitext(img_name)
    print("====================================")
    print(f"Processing: {img_name}")
    print(f"  Watermark text: {WATERMARK_TEXT}")

    sig_bytes = sign_message(WATERMARK_TEXT)
    sig_bits_len = len(bytes_to_bits(sig_bytes))
    print(f"  Signature bytes length: {len(sig_bytes)}")
    print(f"  Signature bits length: {sig_bits_len}")

    img = read_cv_image_3ch(img_path)
    if img is None:
        print("  Error: cannot read image.")
        return False, False, False

    out_dir = os.path.join(ENCODED_BASE_DIR, name_no_ext)
    os.makedirs(out_dir, exist_ok=True)
    comp_dir = os.path.join(COMPARISON_BASE_DIR, name_no_ext)
    os.makedirs(comp_dir, exist_ok=True)
    dec_dir = os.path.join(DECODED_BASE_DIR, name_no_ext)
    os.makedirs(dec_dir, exist_ok=True)

    lsb_ok = dct_ok = dwt_ok = False
    lsb_valid = dct_valid = dwt_valid = False
    lsb_img = dct_img = dwt_img = None

    ok, wm = encode_lsb(img, sig_bytes)
    if ok:
        lsb_ok = True
        lsb_img = wm
        cv2.imwrite(os.path.join(out_dir, "LSB.png"), lsb_img)

    ok, wm = encode_dct(img, sig_bytes)
    if not ok:
        print("  Error: Message too large to encode in image (DCT)")
    else:
        dct_ok = True
        dct_img = wm
        cv2.imwrite(os.path.join(out_dir, "DCT.png"), dct_img)

    ok, wm = encode_dwt(img, sig_bytes)
    if not ok:
        print("  Error: Message too large to encode in image (DWT)")
    else:
        dwt_ok = True
        dwt_img = wm
        cv2.imwrite(os.path.join(out_dir, "DWT.png"), dwt_img)

    if lsb_ok:
        lsb_dec_bytes = decode_lsb(lsb_img)
        lsb_dec_hex = binascii.hexlify(lsb_dec_bytes).decode("ascii")
        save_text(os.path.join(dec_dir, "LSB_decoded.txt"), lsb_dec_hex)
        lsb_valid = verify_signature(WATERMARK_TEXT, lsb_dec_bytes)

    if dct_ok:
        dct_dec_bytes = decode_dct(dct_img, sig_bits_len)
        dct_dec_hex = binascii.hexlify(dct_dec_bytes).decode("ascii")
        save_text(os.path.join(dec_dir, "DCT_decoded.txt"), dct_dec_hex)
        dct_valid = verify_signature(WATERMARK_TEXT, dct_dec_bytes)

    if dwt_ok:
        dwt_dec_bytes = decode_dwt(dwt_img, sig_bits_len)
        dwt_dec_hex = binascii.hexlify(dwt_dec_bytes).decode("ascii")
        save_text(os.path.join(dec_dir, "DWT_decoded.txt"), dwt_dec_hex)
        dwt_valid = verify_signature(WATERMARK_TEXT, dwt_dec_bytes)

    print(f"  LSB watermark valid: {lsb_valid}")
    print(f"  DCT watermark valid: {dct_valid}")
    print(f"  DWT watermark valid: {dwt_valid}")

    wb = Workbook()
    ws = wb.active
    ws.title = "Comparison"

    headers = ["Image", "LSB watermark valid", "DCT watermark valid", "DWT watermark valid"]
    ws.append(headers)
    ws.append([img_name, str(lsb_valid), str(dct_valid), str(dwt_valid)])

    wb.save(os.path.join(comp_dir, "Comparison.xlsx"))
    return lsb_valid, dct_valid, dwt_valid


def main():
    ensure_keys()
    if not os.path.isdir(ORIGINAL_DIR):
        print(f"Original image folder not found: {ORIGINAL_DIR}")
        return
    files = sorted(
        f for f in os.listdir(ORIGINAL_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )
    if not files:
        print("No images found in Original_image folder.")
        return

    lsb_false_list = []
    dct_false_list = []
    dwt_false_list = []

    for fname in files:
        path = os.path.join(ORIGINAL_DIR, fname)
        lsb_valid, dct_valid, dwt_valid = process_one_image(path)
        if not lsb_valid:
            lsb_false_list.append(fname)
        if not dct_valid:
            dct_false_list.append(fname)
        if not dwt_valid:
            dwt_false_list.append(fname)

    print("====================================")
    print("LSB false count:", len(lsb_false_list),
          "False:", ", ".join(lsb_false_list) if lsb_false_list else "None")
    print("DCT false count:", len(dct_false_list),
          "False:", ", ".join(dct_false_list) if dct_false_list else "None")
    print("DWT false count:", len(dwt_false_list),
          "False:", ", ".join(dwt_false_list) if dwt_false_list else "None")
    print("All images processed.")


if __name__ == "__main__":
    main()
