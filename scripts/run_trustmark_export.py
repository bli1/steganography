import sys
sys.path.append("/Users/Xinlei/Desktop/科研2/trustmark/python")

from pathlib import Path
import base64
import binascii
from PIL import Image

from trustmark import TrustMark

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "Original_image"
BASE_OUT_DIR = ROOT / "TrustMark_encoded"

PRIV_KEY_PATH = ROOT / "private_key.pem"
PUB_KEY_PATH = ROOT / "public_key.pem"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

WATERMARK_TEXT = "ASys Encryption"


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        data = f.read()
    return serialization.load_pem_private_key(data, password=None)


def sign_message(msg: str) -> bytes:
    private_key = load_private_key()
    sig = private_key.sign(
        msg.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return sig


def make_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    i = 1
    while True:
        cand = base_dir.parent / f"{base_dir.name}_{i}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        i += 1


def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def build_secret_from_signature(sig_bytes: bytes) -> str:
    digest = hashes.Hash(hashes.SHA256())
    digest.update(sig_bytes)
    h = digest.finalize()
    secret8 = base64.b32encode(h[:5]).decode("ascii").rstrip("=")
    return secret8


def main():
    print("ROOT =", ROOT)
    print("INPUT_DIR =", INPUT_DIR)
    print("PRIV_KEY_PATH =", PRIV_KEY_PATH)

    if not INPUT_DIR.exists():
        raise RuntimeError(f"找不到输入目录: {INPUT_DIR.resolve()}")

    if not PRIV_KEY_PATH.exists():
        raise RuntimeError(
            f"找不到 private_key.pem: {PRIV_KEY_PATH.resolve()}\n"
            "请确认 private_key.pem 在科研2根目录。"
        )

    images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])

    if len(images) == 0:
        raise RuntimeError(f"输入目录里没有找到图片: {INPUT_DIR.resolve()}")

    print(f"[OK] 发现 {len(images)} 张图片来自 {INPUT_DIR.resolve()}")

    private_key = load_private_key()
    key_bits = getattr(private_key, "key_size", None)

    sig_bytes = private_key.sign(
        WATERMARK_TEXT.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    secret = build_secret_from_signature(sig_bytes)

    print(f"[OK] WATERMARK_TEXT = {WATERMARK_TEXT}")
    if key_bits is not None:
        print(f"[OK] RSA key size = {key_bits} bits")
    print(f"[OK] signature bytes = {len(sig_bytes)}")
    print(f"[OK] secret (commit) = {secret}  (len={len(secret)} chars)")

    sig_hex = binascii.hexlify(sig_bytes).decode("ascii")

    out_dir = make_unique_dir(BASE_OUT_DIR)
    print(f"[OK] 输出目录: {out_dir.resolve()}")

    tm = TrustMark()

    ok_cnt = 0
    warn_cnt = 0

    for p in images:
        img = load_rgb(p)

        try:
            wm_img = tm.encode(img, payload=secret)
        except TypeError:
            wm_img = tm.encode(img, secret)
        except Exception as e:
            raise RuntimeError(f"[ERR] TrustMark.encode 失败: {p.name} -> {e}")

        out_path = out_dir / p.name
        wm_img.save(out_path)

        try:
            wm_secret, wm_present, wm_schema = tm.decode(wm_img)
        except Exception as e:
            wm_secret, wm_present, wm_schema = None, False, None
            print(f"[WARN] decode 异常: {p.name} -> {e}")

        if (wm_present is True) and (wm_secret == secret):
            ok_cnt += 1
            print(f"  [OK] {p.name} -> {out_path.name} | decoded={wm_secret} | schema={wm_schema}")
        else:
            warn_cnt += 1
            print(f"  [WARN] {p.name} -> {out_path.name} | present={wm_present} decoded={wm_secret} schema={wm_schema}")

    print(f"[DONE] TrustMark embedding finished. OK={ok_cnt}, WARN={warn_cnt}")
    if warn_cnt > 0:
        print("[HINT] 若 WARN 很多，请检查 trustmark 版本、依赖、图片模式，或尝试调整参数。")


if __name__ == "__main__":
    main()
