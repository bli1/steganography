import base64
import hashlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pywt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = ROOT / "Original_image"
CLEAN_DIR = ROOT / "All_geo_encoded"
GEO_DIR = ROOT / "All_geo_attacked"
PRIVATE_KEY_PATH = ROOT / "private_key.pem"
PUBLIC_KEY_PATH = ROOT / "public_key.pem"

WATERMARK_TEXT = "ASys Encryption"
SCALE = 0.8
DCT_Q = 8.0
DWT_Q = 8.0
SS_ALPHA = 20.0
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_rsa2048():
    if PRIVATE_KEY_PATH.exists() and PUBLIC_KEY_PATH.exists():
        try:
            private = serialization.load_pem_private_key(PRIVATE_KEY_PATH.read_bytes(), password=None)
            public = serialization.load_pem_public_key(PUBLIC_KEY_PATH.read_bytes())
            if private.key_size != 2048 or public.key_size != 2048:
                raise ValueError("existing keys are not RSA-2048")
            test = private.sign(WATERMARK_TEXT.encode(), padding.PKCS1v15(), hashes.SHA256())
            public.verify(test, WATERMARK_TEXT.encode(), padding.PKCS1v15(), hashes.SHA256())
            return private, public
        except Exception as exc:
            raise RuntimeError(
                f"Invalid RSA key pair: {exc}. Do not overwrite experiment keys silently. "
                "Move the two PEM files away, then rerun to generate one RSA-2048 pair."
            ) from exc

    if PRIVATE_KEY_PATH.exists() != PUBLIC_KEY_PATH.exists():
        raise RuntimeError("Only one PEM key exists. Restore the matching key or remove both files.")

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key()
    PRIVATE_KEY_PATH.write_bytes(private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    PUBLIC_KEY_PATH.write_bytes(public.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    return private, public


def signature_and_token(private, public):
    signature = private.sign(WATERMARK_TEXT.encode(), padding.PKCS1v15(), hashes.SHA256())
    public.verify(signature, WATERMARK_TEXT.encode(), padding.PKCS1v15(), hashes.SHA256())
    if len(signature) != 256:
        raise RuntimeError(f"Expected a 256-byte RSA-2048 signature, got {len(signature)} bytes")
    token = base64.b32encode(hashlib.sha256(signature).digest()).decode("ascii")[:8]
    return signature, token


def read_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img[:, :, :3]


def bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def encode_lsb(img, data):
    payload = bits(data)
    header = np.array([int(x) for x in f"{payload.size:032b}"], dtype=np.uint8)
    all_bits = np.concatenate((header, payload))
    flat = img.reshape(-1, 3).copy()
    if all_bits.size > flat.shape[0]:
        raise ValueError(f"LSB capacity {flat.shape[0]} bits < required {all_bits.size} bits")
    flat[:all_bits.size, 0] = (flat[:all_bits.size, 0] & 0xFE) | all_bits
    return flat.reshape(img.shape)


def encode_dct(img, data):
    payload = bits(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    positions = [(4, 4), (3, 3), (2, 2), (5, 5)]
    capacity = (h // 8) * (w // 8) * len(positions)
    if payload.size > capacity:
        raise ValueError(f"DCT capacity {capacity} bits < required {payload.size} bits")
    work = gray.astype(np.float32)
    k = 0
    for y in range(0, h - 7, 8):
        for x in range(0, w - 7, 8):
            block = cv2.dct(work[y:y + 8, x:x + 8])
            for ry, rx in positions:
                if k >= payload.size:
                    break
                q = int(np.round(block[ry, rx] / DCT_Q))
                if (q & 1) != int(payload[k]):
                    q += 1 if block[ry, rx] >= 0 else -1
                block[ry, rx] = q * DCT_Q
                k += 1
            work[y:y + 8, x:x + 8] = cv2.idct(block)
            if k >= payload.size:
                break
        if k >= payload.size:
            break
    out = np.clip(work, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def encode_dwt(img, data):
    payload = bits(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ll, (lh, hl, hh) = pywt.dwt2(gray, "haar")
    flat = lh.ravel().copy()
    if payload.size > flat.size:
        raise ValueError(f"DWT capacity {flat.size} bits < required {payload.size} bits")
    for i, bit in enumerate(payload):
        q = int(np.round(flat[i] / DWT_Q))
        if (q & 1) != int(bit):
            q += 1 if flat[i] >= 0 else -1
        flat[i] = q * DWT_Q
    rec = pywt.idwt2((ll, (flat.reshape(lh.shape), hl, hh)), "haar")
    rec = np.clip(rec[:gray.shape[0], :gray.shape[1]], 0, 255).astype(np.uint8)
    return cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR)


def seed_base():
    return int(hashlib.sha256(WATERMARK_TEXT.encode()).hexdigest(), 16) % (2**31 - 1)


def spread_embed(flat, payload):
    if flat.size < payload.size:
        raise ValueError(f"Spread-spectrum capacity {flat.size} < required {payload.size}")
    chunk = flat.size // payload.size
    out = flat.copy()
    base = seed_base()
    for i, bit in enumerate(payload):
        start = i * chunk
        end = flat.size if i == payload.size - 1 else (i + 1) * chunk
        pn = np.random.RandomState(base + i).choice([-1, 1], end - start).astype(np.float32)
        out[start:end] += SS_ALPHA * (1 if bit else -1) * pn
    return out


def encode_spatial_ss(img, data):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    marked = spread_embed(gray.ravel(), bits(data)).reshape(gray.shape)
    return cv2.cvtColor(np.clip(marked, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def encode_dwt_ss(img, data):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ll, (lh, hl, hh) = pywt.dwt2(gray, "haar")
    marked = spread_embed(lh.ravel(), bits(data)).reshape(lh.shape)
    rec = pywt.idwt2((ll, (marked, hl, hh)), "haar")
    rec = np.clip(rec[:gray.shape[0], :gray.shape[1]], 0, 255).astype(np.uint8)
    return cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR)


def scale_attack(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * SCALE)), max(1, int(h * SCALE))),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def get_trustmark():
    try:
        from trustmark import TrustMark
    except ImportError as exc:
        raise RuntimeError("TrustMark is not installed. Run: pip install trustmark") from exc
    try:
        return TrustMark(verbose=False, model_type="Q", use_ECC=True)
    except TypeError:
        return TrustMark(verbose=False, model_type="Q")


def trustmark_encode(tm, img, token):
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = tm.encode(pil, token)
    if isinstance(result, tuple):
        result = result[0]
    return cv2.cvtColor(np.asarray(result.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    start = time.time()
    private, public = ensure_rsa2048()
    signature, token = signature_and_token(private, public)
    print("=" * 72)
    print("ALL METHODS: GENERATE + SCALE ATTACK")
    print("RSA key size      : 2048 bits")
    print("RSA signature     : 256 bytes / 2048 bits")
    print("TrustMark token   :", token)
    print("Scale attack      :", SCALE)
    print("=" * 72)

    if not ORIGINAL_DIR.is_dir():
        print("Original_image folder not found:", ORIGINAL_DIR)
        return 1
    files = sorted(p for p in ORIGINAL_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        print("No images found in:", ORIGINAL_DIR)
        return 1

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    tm = get_trustmark()
    methods = ("LSB", "DCT", "DWT", "SpatialSS", "DWTSS", "TrustMark")
    successes = {m: 0 for m in methods}
    failures = []

    for index, path in enumerate(files, 1):
        relative = path.relative_to(ORIGINAL_DIR)
        item_id = str(relative.with_suffix(""))
        clean_out = CLEAN_DIR / item_id
        geo_out = GEO_DIR / item_id
        clean_out.mkdir(parents=True, exist_ok=True)
        geo_out.mkdir(parents=True, exist_ok=True)
        img = read_bgr(path)
        if img is None:
            failures.append((item_id, "ALL", "cannot read image"))
            continue
        print(f"[{index}/{len(files)}] {relative}")
        encoders = {
            "LSB": lambda: encode_lsb(img, signature),
            "DCT": lambda: encode_dct(img, signature),
            "DWT": lambda: encode_dwt(img, signature),
            "SpatialSS": lambda: encode_spatial_ss(img, signature),
            "DWTSS": lambda: encode_dwt_ss(img, signature),
            "TrustMark": lambda: trustmark_encode(tm, img, token),
        }
        for method, encoder in encoders.items():
            try:
                marked = encoder()
                ok1 = cv2.imwrite(str(clean_out / f"{method}.png"), marked)
                ok2 = cv2.imwrite(str(geo_out / f"{method}_geo.png"), scale_attack(marked))
                if not (ok1 and ok2):
                    raise OSError("cv2.imwrite returned False")
                successes[method] += 1
            except Exception as exc:
                failures.append((item_id, method, str(exc)))
                print(f"  [FAIL] {method}: {exc}")

    log_path = GEO_DIR / "generation_failures.tsv"
    log_path.write_text("Image\tMethod\tError\n" + "\n".join("\t".join(x) for x in failures), encoding="utf-8")
    print("=" * 72)
    print("Images found      :", len(files))
    for method in methods:
        print(f"{method:<18}: {successes[method]}/{len(files)}")
    print("Failed operations :", len(failures))
    print("Runtime           : %.2f seconds" % (time.time() - start))
    print("Clean output      :", CLEAN_DIR)
    print("Attacked output   :", GEO_DIR)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
