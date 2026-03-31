from pathlib import Path
import csv
import base64
from PIL import Image
from trustmark import TrustMark
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

ROOT = Path("..")
JPEG_DIR = ROOT / "TrustMark_jpeg"
OUT_DIR = ROOT / "TrustMark_jpeg_result"
PRIV_KEY_PATH = ROOT / "private_key.pem"
WATERMARK_TEXT = "ASys Encryption"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def build_secret_from_signature(sig_bytes: bytes) -> str:
    digest = hashes.Hash(hashes.SHA256())
    digest.update(sig_bytes)
    h = digest.finalize()
    return base64.b32encode(h[:5]).decode("ascii").rstrip("=")

def expected_secret():
    sk = load_private_key()
    sig = sk.sign(WATERMARK_TEXT.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
    return build_secret_from_signature(sig)

def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def list_quality_dirs():
    if not JPEG_DIR.exists():
        return []
    out = []
    for d in JPEG_DIR.iterdir():
        if d.is_dir() and d.name.startswith("Q"):
            try:
                q = int(d.name[1:])
                out.append((q, d))
            except Exception:
                pass
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def main():
    if not PRIV_KEY_PATH.exists():
        raise RuntimeError(f"private_key.pem not found: {PRIV_KEY_PATH.resolve()}")
    q_dirs = list_quality_dirs()
    if not q_dirs:
        print("No JPEG folders found in:", JPEG_DIR.resolve())
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_image_csv = OUT_DIR / "per_image_trustmark_jpeg.csv"
    summary_csv = OUT_DIR / "summary_trustmark_jpeg.csv"
    exp = expected_secret()
    tm = TrustMark()
    per_rows = []
    sum_rows = []
    for q, q_path in q_dirs:
        imgs = sorted([p for p in q_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        total = 0
        present_cnt = 0
        match_cnt = 0
        err_cnt = 0
        for p in imgs:
            total += 1
            try:
                img = load_rgb(p)
                secret, present, schema = tm.decode(img)
                if present:
                    present_cnt += 1
                match = bool(present and secret == exp)
                if match:
                    match_cnt += 1
                per_rows.append([q, p.name, bool(present), str(secret) if secret is not None else "", str(schema) if schema is not None else "", match, ""])
            except Exception as e:
                err_cnt += 1
                per_rows.append([q, p.name, False, "", "", False, str(e)])
        present_rate = (present_cnt / total) if total else 0.0
        match_rate = (match_cnt / total) if total else 0.0
        sum_rows.append([q, total, present_cnt, present_rate, match_cnt, match_rate, err_cnt])
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jpeg_quality", "image", "present", "secret", "schema", "match_expected", "error"])
        w.writerows(per_rows)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jpeg_quality", "total_images", "present_count", "present_rate", "match_count", "match_rate", "error_count"])
        w.writerows(sum_rows)
    print("Saved:", per_image_csv.resolve())
    print("Saved:", summary_csv.resolve())

if __name__ == "__main__":
    main()
