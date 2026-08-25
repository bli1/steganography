# trustmark_detect_rsa2048.py
# ============================================================
# TrustMark + RSA-2048 Detection and Authentication
#
# 脚本位置：
# steganography-main\scripts\trustmark_detect_rsa2048.py
#
# 项目结构：
# steganography-main
# ├─ private_key.pem
# ├─ public_key.pem
# ├─ TrustMark_encoded
# ├─ TrustMark_blur
# └─ scripts
#    └─ trustmark_detect_rsa2048.py
# ============================================================

from pathlib import Path
import base64
import csv

from PIL import Image
from trustmark import TrustMark

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# ============================================================
# 路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PRIVATE_KEY_PATH = ROOT / "private_key.pem"
PUBLIC_KEY_PATH = ROOT / "public_key.pem"


# ============================================================
# 选择检测文件夹
# ============================================================

# 未攻击的TrustMark图片：
DETECT_DIR_NAME = "TrustMark_blur"

# 检测Gaussian Blur攻击图片时改为：
# DETECT_DIR_NAME = "TrustMark_blur"

# 检测JPEG攻击图片时改为：
# DETECT_DIR_NAME = "TrustMark_jpeg"

DETECT_DIR = ROOT / DETECT_DIR_NAME


# ============================================================
# RSA和水印参数
# 必须与run_trustmark_export.py完全一致
# ============================================================

WATERMARK_TEXT = "ASys Encryption"
REQUIRED_RSA_BITS = 2048

IMG_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# RSA工具
# ============================================================

def load_private_key():
    """读取项目根目录中的外部RSA私钥。"""
    with open(PRIVATE_KEY_PATH, "rb") as key_file:
        return serialization.load_pem_private_key(
            key_file.read(),
            password=None,
        )


def load_public_key():
    """读取项目根目录中的外部RSA公钥。"""
    with open(PUBLIC_KEY_PATH, "rb") as key_file:
        return serialization.load_pem_public_key(
            key_file.read()
        )


def sign_message(private_key, message: str) -> bytes:
    """
    使用外部RSA-2048私钥签名。

    PKCS1v15签名对相同密钥、相同文字和相同哈希算法
    会产生相同签名。
    """
    return private_key.sign(
        message.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def verify_signature(
    public_key,
    message: str,
    signature: bytes,
) -> bool:
    """使用外部RSA-2048公钥验证签名。"""
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


def build_commit_from_signature(signature: bytes) -> str:
    """
    根据RSA签名生成TrustMark payload。

    必须与嵌入代码保持一致：

    RSA signature
        → SHA-256
        → 取前5字节（40 bits）
        → Base32
        → 8字符TrustMark payload
    """
    digest = hashes.Hash(hashes.SHA256())
    digest.update(signature)
    signature_hash = digest.finalize()

    return base64.b32encode(
        signature_hash[:5]
    ).decode("ascii").rstrip("=")


# ============================================================
# 图片工具
# ============================================================

def list_images(folder: Path):
    """获取检测文件夹中的全部图片。"""
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMG_EXTS
        ],
        key=lambda path: path.name.lower(),
    )


def load_rgb(image_path: Path) -> Image.Image:
    """读取图片并转换成RGB格式。"""
    with Image.open(image_path) as image:
        return image.convert("RGB")


# ============================================================
# TrustMark检测
# ============================================================

def detect_one(
    trustmark_model: TrustMark,
    image_path: Path,
):
    """
    检测一张图片。

    返回：
        decoded_payload
        watermark_present
        schema
        error_message
    """
    try:
        image = load_rgb(image_path)

        decoded_payload, watermark_present, schema = (
            trustmark_model.decode(image)
        )

        return (
            decoded_payload,
            bool(watermark_present),
            schema,
            None,
        )

    except Exception as error:
        return None, False, None, str(error)


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 72)
    print("TRUSTMARK + RSA-2048 DETECTION")
    print("=" * 72)

    print(f"Project root       : {ROOT.resolve()}")
    print(f"Detection folder   : {DETECT_DIR.resolve()}")
    print(f"Private key        : {PRIVATE_KEY_PATH.resolve()}")
    print(f"Public key         : {PUBLIC_KEY_PATH.resolve()}")
    print()

    # --------------------------------------------------------
    # 检查文件
    # --------------------------------------------------------

    if not DETECT_DIR.is_dir():
        raise RuntimeError(
            f"找不到检测目录：\n"
            f"{DETECT_DIR.resolve()}"
        )

    if not PRIVATE_KEY_PATH.is_file():
        raise RuntimeError(
            f"找不到RSA私钥：\n"
            f"{PRIVATE_KEY_PATH.resolve()}"
        )

    if not PUBLIC_KEY_PATH.is_file():
        raise RuntimeError(
            f"找不到RSA公钥：\n"
            f"{PUBLIC_KEY_PATH.resolve()}"
        )

    images = list_images(DETECT_DIR)

    if not images:
        raise RuntimeError(
            f"检测目录中没有图片：\n"
            f"{DETECT_DIR.resolve()}"
        )

    total_images = len(images)

    print(f"[OK] Images found = {total_images}")

    # --------------------------------------------------------
    # 加载外部RSA密钥
    # --------------------------------------------------------

    private_key = load_private_key()
    public_key = load_public_key()

    private_key_bits = getattr(
        private_key,
        "key_size",
        None,
    )

    public_key_bits = getattr(
        public_key,
        "key_size",
        None,
    )

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

    print(
        f"[OK] RSA private key size = "
        f"{private_key_bits} bits"
    )

    print(
        f"[OK] RSA public key size  = "
        f"{public_key_bits} bits"
    )

    # --------------------------------------------------------
    # 使用外部私钥重新生成相同签名
    # --------------------------------------------------------

    signature = sign_message(
        private_key,
        WATERMARK_TEXT,
    )

    print(
        f"[OK] RSA signature size   = "
        f"{len(signature)} bytes"
    )

    # --------------------------------------------------------
    # 使用外部公钥验证签名
    # --------------------------------------------------------

    rsa_signature_verified = verify_signature(
        public_key,
        WATERMARK_TEXT,
        signature,
    )

    if not rsa_signature_verified:
        raise RuntimeError(
            "RSA签名验证失败。\n"
            "private_key.pem和public_key.pem可能不是同一对密钥，"
            "或者WATERMARK_TEXT不一致。"
        )

    print("[OK] RSA signature verification = PASS")

    # --------------------------------------------------------
    # 自动计算期望payload
    # --------------------------------------------------------

    expected_payload = build_commit_from_signature(
        signature
    )

    print(f"[OK] Watermark text   = {WATERMARK_TEXT}")
    print(f"[OK] Expected payload = {expected_payload}")
    print()

    # --------------------------------------------------------
    # 初始化TrustMark
    # --------------------------------------------------------

    print("[INFO] Initializing TrustMark model...")

    try:
        trustmark_model = TrustMark()

    except Exception as error:
        raise RuntimeError(
            f"TrustMark模型初始化失败：{error}"
        ) from error

    print("[OK] TrustMark model initialized.")
    print()

    # --------------------------------------------------------
    # 检测统计
    # --------------------------------------------------------

    detected_count = 0
    exact_match_count = 0
    authenticated_count = 0
    decode_failed_count = 0

    results = []

    # --------------------------------------------------------
    # 逐张检测
    # --------------------------------------------------------

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        (
            decoded_payload,
            watermark_present,
            schema,
            error_message,
        ) = detect_one(
            trustmark_model,
            image_path,
        )

        if error_message is not None:
            decode_failed_count += 1

            results.append(
                {
                    "folder": DETECT_DIR_NAME,
                    "image": image_path.name,
                    "watermark_present": False,
                    "decoded_payload": "",
                    "expected_payload": expected_payload,
                    "payload_matched": False,
                    "rsa_key_bits": REQUIRED_RSA_BITS,
                    "rsa_signature_verified": (
                        rsa_signature_verified
                    ),
                    "authenticated": False,
                    "schema": "",
                    "error": error_message,
                }
            )

            print(
                f"[{index}/{total_images}] "
                f"[DECODE ERROR] {image_path.name}: "
                f"{error_message}"
            )

            continue

        if watermark_present:
            detected_count += 1

        payload_matched = bool(
            watermark_present
            and decoded_payload == expected_payload
        )

        if payload_matched:
            exact_match_count += 1

        # 图片认证成功必须同时满足：
        # 1. RSA签名通过公钥验证
        # 2. 图片存在TrustMark
        # 3. 解出的payload与RSA签名commit一致
        authenticated = bool(
            rsa_signature_verified
            and watermark_present
            and payload_matched
        )

        if authenticated:
            authenticated_count += 1

        results.append(
            {
                "folder": DETECT_DIR_NAME,
                "image": image_path.name,
                "watermark_present": watermark_present,
                "decoded_payload": (
                    decoded_payload
                    if decoded_payload is not None
                    else ""
                ),
                "expected_payload": expected_payload,
                "payload_matched": payload_matched,
                "rsa_key_bits": REQUIRED_RSA_BITS,
                "rsa_signature_verified": (
                    rsa_signature_verified
                ),
                "authenticated": authenticated,
                "schema": (
                    schema
                    if schema is not None
                    else ""
                ),
                "error": "",
            }
        )

        # 每50张显示一次进度
        if (
            index == 1
            or index % 50 == 0
            or index == total_images
        ):
            print(
                f"[{index}/{total_images}] "
                f"Detected={detected_count}, "
                f"Matched={exact_match_count}, "
                f"Authenticated={authenticated_count}, "
                f"Failed={decode_failed_count}"
            )

    # --------------------------------------------------------
    # 计算比率
    # --------------------------------------------------------

    detection_rate = (
        detected_count / total_images
    )

    exact_match_rate = (
        exact_match_count / total_images
    )

    authentication_rate = (
        authenticated_count / total_images
    )

    decode_failure_rate = (
        decode_failed_count / total_images
    )

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TRUSTMARK + RSA-2048 SUMMARY")
    print("=" * 72)

    print(f"Folder                 : {DETECT_DIR_NAME}")
    print(f"Total images           : {total_images}")
    print(
        f"RSA key size           : "
        f"{REQUIRED_RSA_BITS} bits"
    )
    print(
        f"RSA signature verified : "
        f"{rsa_signature_verified}"
    )
    print(f"Expected payload       : {expected_payload}")

    print(
        f"Decode failed          : "
        f"{decode_failed_count}/{total_images}"
    )

    print(
        f"Watermark detected     : "
        f"{detected_count}/{total_images}"
    )

    print(
        f"Exact payload matched  : "
        f"{exact_match_count}/{total_images}"
    )

    print(
        f"Authenticated          : "
        f"{authenticated_count}/{total_images}"
    )

    print(f"Detection rate         : {detection_rate:.6f}")
    print(f"Exact-match rate       : {exact_match_rate:.6f}")
    print(f"Authentication rate    : {authentication_rate:.6f}")
    print(f"Decode-failure rate    : {decode_failure_rate:.6f}")

    # --------------------------------------------------------
    # 保存逐图CSV
    # --------------------------------------------------------

    output_csv = ROOT / (
        f"TrustMark_RSA2048_detect_{DETECT_DIR_NAME}.csv"
    )

    fieldnames = [
        "folder",
        "image",
        "watermark_present",
        "decoded_payload",
        "expected_payload",
        "payload_matched",
        "rsa_key_bits",
        "rsa_signature_verified",
        "authenticated",
        "schema",
        "error",
    ]

    try:
        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(results)

        print(f"[OK] CSV saved: {output_csv.resolve()}")

    except Exception as error:
        print(f"[WARN] CSV保存失败：{error}")

    # --------------------------------------------------------
    # 保存汇总CSV
    # --------------------------------------------------------

    summary_csv = ROOT / (
        f"TrustMark_RSA2048_summary_{DETECT_DIR_NAME}.csv"
    )

    try:
        with open(
            summary_csv,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(
                [
                    "folder",
                    "total_images",
                    "rsa_key_bits",
                    "rsa_signature_verified",
                    "expected_payload",
                    "decode_failed",
                    "watermark_detected",
                    "exact_payload_matched",
                    "authenticated",
                    "detection_rate",
                    "exact_match_rate",
                    "authentication_rate",
                    "decode_failure_rate",
                ]
            )

            writer.writerow(
                [
                    DETECT_DIR_NAME,
                    total_images,
                    REQUIRED_RSA_BITS,
                    rsa_signature_verified,
                    expected_payload,
                    decode_failed_count,
                    detected_count,
                    exact_match_count,
                    authenticated_count,
                    detection_rate,
                    exact_match_rate,
                    authentication_rate,
                    decode_failure_rate,
                ]
            )

        print(f"[OK] Summary CSV saved: {summary_csv.resolve()}")

    except Exception as error:
        print(f"[WARN] Summary CSV保存失败：{error}")

    print("=" * 72)
    print("[DONE] TrustMark RSA-2048 detection finished.")


if __name__ == "__main__":
    main()