from pathlib import Path
import base64
import binascii
from PIL import Image

from trustmark import TrustMark

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


ROOT = Path("..")
INPUT_DIR = ROOT / "Original_image"
BASE_OUT_DIR = ROOT / "TrustMark_encoded"

PRIV_KEY_PATH = ROOT / "private_key.pem"
PUB_KEY_PATH = ROOT / "public_key.pem"  # 这里暂时不用，但保留一致性

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
    """避免覆盖：TrustMark_encoded, TrustMark_encoded_1, ..."""
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
    """
    ✅ TrustMark payload 容量非常小（100 bits 总容量，扣掉ECC后可用更少），
    不能直接塞 RSA 签名hex（太长）。
    这里把 signature 做 SHA256，再取前 5 bytes = 40 bits，
    用 base32 编码成固定 8 个字符的 commit（稳定、短、可追溯）。
    """
    digest = hashes.Hash(hashes.SHA256())
    digest.update(sig_bytes)
    h = digest.finalize()  # 32 bytes

    # 40 bits -> base32 刚好 8 chars（无 '='）
    secret8 = base64.b32encode(h[:5]).decode("ascii").rstrip("=")
    return secret8


def main():
    # 1) 检查输入
    if not INPUT_DIR.exists():
        raise RuntimeError(f"找不到输入目录: {INPUT_DIR.resolve()}")

    if not PRIV_KEY_PATH.exists():
        raise RuntimeError(
            f"找不到 private_key.pem: {PRIV_KEY_PATH.resolve()}\n"
            "请确认你原来的脚本生成的 private_key.pem 在科研2根目录。"
        )

    images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"[OK] 发现 {len(images)} 张图片来自 {INPUT_DIR.resolve()}")

    # 2) 生成 payload：先签名，再压缩成短 commit
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

    # （可选）如果你还想保留“签名hex”用于日志/溯源存档（不嵌入水印）
    sig_hex = binascii.hexlify(sig_bytes).decode("ascii")
    # print(f"[DBG] sig_hex length = {len(sig_hex)}")

    # 3) 输出目录
    out_dir = make_unique_dir(BASE_OUT_DIR)
    print(f"[OK] 输出目录: {out_dir.resolve()}")

    # 4) TrustMark encode + sanity decode
    tm = TrustMark()

    ok_cnt = 0
    warn_cnt = 0

    for p in images:
        img = load_rgb(p)

        try:
            # 显式使用 payload=，避免不同版本参数位置差异
            wm_img = tm.encode(img, payload=secret)
        except TypeError:
            # 如果你的 trustmark 版本不支持关键字参数，就回退到 positional
            wm_img = tm.encode(img, secret)
        except Exception as e:
            raise RuntimeError(f"[ERR] TrustMark.encode 失败: {p.name} -> {e}")

        # 保存
        out_path = out_dir / p.name
        wm_img.save(out_path)

        # ✅ sanity check：立即 decode 验证无攻击情况下可读
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
        print("[HINT] 若 WARN 很多：请检查 trustmark 版本/依赖、图片模式、或尝试调整 TrustMark 参数（如 WM_STRENGTH / ECC）。")


if __name__ == "__main__":
    main()
