from pathlib import Path
import base64

from PIL import Image
from trustmark import TrustMark

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# ============================================================
# 路径设置
# 当前脚本应放在：
# steganography-main\scripts\trustmark_encode.py
#
# 项目目录结构：
# steganography-main
# ├─ Original_image
# ├─ private_key.pem
# ├─ public_key.pem
# └─ scripts
#    └─ trustmark_encode.py
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "Original_image"
BASE_OUT_DIR = ROOT / "TrustMark_encoded"

PRIV_KEY_PATH = ROOT / "private_key.pem"
PUB_KEY_PATH = ROOT / "public_key.pem"


# ============================================================
# 实验参数
# ============================================================

IMG_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

WATERMARK_TEXT = "ASys Encryption"

# 要求使用RSA-2048
REQUIRED_RSA_BITS = 2048


# ============================================================
# RSA工具
# ============================================================

def load_private_key():
    """读取RSA私钥。"""
    with open(PRIV_KEY_PATH, "rb") as f:
        key_data = f.read()

    return serialization.load_pem_private_key(
        key_data,
        password=None,
    )


def load_public_key():
    """读取RSA公钥。"""
    with open(PUB_KEY_PATH, "rb") as f:
        key_data = f.read()

    return serialization.load_pem_public_key(key_data)


def sign_message(private_key, message: str) -> bytes:
    """使用RSA私钥对指定文字进行签名。"""
    return private_key.sign(
        message.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def verify_signature(public_key, message: str, signature: bytes) -> bool:
    """使用RSA公钥验证签名。"""
    try:
        public_key.verify(
            signature,
            message.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True

    except Exception:
        return False


# ============================================================
# TrustMark payload生成
# ============================================================

def build_secret_from_signature(signature: bytes) -> str:
    """
    对RSA签名计算SHA-256，然后取前5字节（40 bits），
    最后进行Base32编码，生成8字符TrustMark payload。

    注意：
    TrustMark中嵌入的是RSA签名的短指纹，
    不是完整的RSA签名。
    """
    digest = hashes.Hash(hashes.SHA256())
    digest.update(signature)
    signature_hash = digest.finalize()

    secret = base64.b32encode(
        signature_hash[:5]
    ).decode("ascii").rstrip("=")

    return secret


# ============================================================
# 文件夹与图片工具
# ============================================================

def make_unique_dir(base_dir: Path) -> Path:
    """
    创建不覆盖旧结果的输出文件夹。

    第一次运行：
        TrustMark_encoded

    再次运行：
        TrustMark_encoded_1
        TrustMark_encoded_2
        ...
    """
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    index = 1

    while True:
        candidate = base_dir.parent / f"{base_dir.name}_{index}"

        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate

        index += 1


def load_rgb(image_path: Path) -> Image.Image:
    """读取图片并统一转换成RGB格式。"""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def save_watermarked_image(image: Image.Image, output_path: Path):
    """
    保存嵌入水印后的图片。

    PNG、BMP、TIFF等格式正常保存。
    如果原图为JPEG，则使用高质量设置，减少保存过程带来的额外损失。
    """
    suffix = output_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        image.save(
            output_path,
            format="JPEG",
            quality=100,
            subsampling=0,
        )

    elif suffix == ".png":
        image.save(
            output_path,
            format="PNG",
            compress_level=1,
        )

    elif suffix == ".webp":
        image.save(
            output_path,
            format="WEBP",
            lossless=True,
            quality=100,
        )

    else:
        image.save(output_path)


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("TRUSTMARK WATERMARK EMBEDDING")
    print("=" * 70)

    print(f"ROOT            = {ROOT}")
    print(f"INPUT_DIR       = {INPUT_DIR}")
    print(f"PRIVATE KEY     = {PRIV_KEY_PATH}")
    print(f"PUBLIC KEY      = {PUB_KEY_PATH}")
    print()

    # --------------------------------------------------------
    # 检查输入目录
    # --------------------------------------------------------

    if not INPUT_DIR.is_dir():
        raise RuntimeError(
            f"找不到输入目录：\n{INPUT_DIR.resolve()}\n\n"
            "请确认 Original_image 文件夹位于 steganography-main 根目录。"
        )

    # --------------------------------------------------------
    # 检查密钥文件
    # --------------------------------------------------------

    if not PRIV_KEY_PATH.is_file():
        raise RuntimeError(
            f"找不到RSA私钥：\n{PRIV_KEY_PATH.resolve()}\n\n"
            "请确认 private_key.pem 位于 steganography-main 根目录。"
        )

    if not PUB_KEY_PATH.is_file():
        raise RuntimeError(
            f"找不到RSA公钥：\n{PUB_KEY_PATH.resolve()}\n\n"
            "请确认 public_key.pem 位于 steganography-main 根目录。"
        )

    # --------------------------------------------------------
    # 搜索图片
    # --------------------------------------------------------

    images = sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMG_EXTS
        ],
        key=lambda path: path.name.lower(),
    )

    if not images:
        raise RuntimeError(
            f"输入目录中没有找到支持的图片：\n"
            f"{INPUT_DIR.resolve()}"
        )

    print(f"[OK] 找到 {len(images)} 张原始图片。")

    # --------------------------------------------------------
    # 加载并检查RSA密钥
    # --------------------------------------------------------

    private_key = load_private_key()
    public_key = load_public_key()

    private_key_bits = getattr(private_key, "key_size", None)
    public_key_bits = getattr(public_key, "key_size", None)

    if private_key_bits != REQUIRED_RSA_BITS:
        raise RuntimeError(
            f"私钥不是RSA-{REQUIRED_RSA_BITS}。\n"
            f"当前私钥长度：{private_key_bits} bits"
        )

    if public_key_bits != REQUIRED_RSA_BITS:
        raise RuntimeError(
            f"公钥不是RSA-{REQUIRED_RSA_BITS}。\n"
            f"当前公钥长度：{public_key_bits} bits"
        )

    print(f"[OK] RSA private key size = {private_key_bits} bits")
    print(f"[OK] RSA public key size  = {public_key_bits} bits")

    # --------------------------------------------------------
    # 生成RSA签名
    # --------------------------------------------------------

    signature = sign_message(
        private_key,
        WATERMARK_TEXT,
    )

    print(f"[OK] Watermark text  = {WATERMARK_TEXT}")
    print(f"[OK] Signature size = {len(signature)} bytes")

    # --------------------------------------------------------
    # 使用公钥验证刚生成的签名
    # --------------------------------------------------------

    rsa_verified = verify_signature(
        public_key,
        WATERMARK_TEXT,
        signature,
    )

    if not rsa_verified:
        raise RuntimeError(
            "RSA签名验证失败。\n"
            "private_key.pem和public_key.pem可能不是同一对密钥。"
        )

    print("[OK] RSA signature verification = PASS")

    # --------------------------------------------------------
    # 从RSA签名生成TrustMark payload
    # --------------------------------------------------------

    expected_secret = build_secret_from_signature(signature)

    print(
        f"[OK] TrustMark payload = {expected_secret} "
        f"(length={len(expected_secret)} characters)"
    )

    # --------------------------------------------------------
    # 创建输出文件夹
    # --------------------------------------------------------

    output_dir = make_unique_dir(BASE_OUT_DIR)

    print(f"[OK] Output directory = {output_dir.resolve()}")
    print()

    # --------------------------------------------------------
    # 初始化TrustMark
    # --------------------------------------------------------

    print("[INFO] 正在初始化TrustMark模型……")

    try:
        trustmark_model = TrustMark()

    except Exception as error:
        raise RuntimeError(
            "TrustMark模型初始化失败：\n"
            f"{error}"
        ) from error

    print("[OK] TrustMark模型初始化完成。")
    print()

    # --------------------------------------------------------
    # 逐张处理图片
    # --------------------------------------------------------

    embedded_count = 0
    detected_count = 0
    exact_count = 0
    failure_count = 0

    total_images = len(images)

    for index, image_path in enumerate(images, start=1):

        print(
            f"[{index}/{total_images}] Processing: "
            f"{image_path.name}"
        )

        try:
            original_image = load_rgb(image_path)

            # 兼容不同TrustMark版本的encode参数名称
            try:
                watermarked_image = trustmark_model.encode(
                    original_image,
                    payload=expected_secret,
                )

            except TypeError:
                watermarked_image = trustmark_model.encode(
                    original_image,
                    expected_secret,
                )

            output_path = output_dir / image_path.name

            save_watermarked_image(
                watermarked_image,
                output_path,
            )

            embedded_count += 1

            # 使用刚保存的图片重新读取并验证。
            # 这样可以把实际保存过程造成的影响也计算进去。
            saved_image = load_rgb(output_path)

            try:
                decoded_secret, watermark_present, watermark_schema = (
                    trustmark_model.decode(saved_image)
                )

            except Exception as decode_error:
                failure_count += 1

                print(
                    f"    [DECODE ERROR] {decode_error}"
                )

                continue

            if bool(watermark_present):
                detected_count += 1

            if (
                bool(watermark_present)
                and decoded_secret == expected_secret
            ):
                exact_count += 1

                print(
                    f"    [OK] present={watermark_present} | "
                    f"decoded={decoded_secret} | "
                    f"schema={watermark_schema}"
                )

            else:
                failure_count += 1

                print(
                    f"    [WARN] present={watermark_present} | "
                    f"decoded={decoded_secret} | "
                    f"expected={expected_secret} | "
                    f"schema={watermark_schema}"
                )

        except Exception as error:
            failure_count += 1

            print(
                f"    [ERROR] {image_path.name}: {error}"
            )

    # --------------------------------------------------------
    # 输出最终统计
    # --------------------------------------------------------

    detection_rate = (
        detected_count / total_images
        if total_images > 0
        else 0.0
    )

    exact_rate = (
        exact_count / total_images
        if total_images > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("TRUSTMARK EMBEDDING SUMMARY")
    print("=" * 70)

    print(f"Total input images : {total_images}")
    print(f"Embedded images    : {embedded_count}")
    print(f"Detected images    : {detected_count}")
    print(f"Exact matches      : {exact_count}")
    print(f"Failures/Warnings  : {failure_count}")
    print(f"Detection rate     : {detection_rate:.6f}")
    print(f"Exact-match rate   : {exact_rate:.6f}")
    print(f"Expected payload   : {expected_secret}")
    print(f"Output directory   : {output_dir.resolve()}")

    print("=" * 70)
    print("[DONE] TrustMark embedding finished.")


if __name__ == "__main__":
    main()