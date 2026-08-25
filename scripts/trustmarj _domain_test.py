import os
import time
import math
import base64
import hashlib

import cv2
import numpy as np
import xlwt
from PIL import Image
from trustmark import TrustMark

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization


# ========================= PATHS =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRUSTMARK_DIR = os.path.join(BASE_DIR, "TrustMark_encoded")
OUTPUT_DIR = os.path.join(BASE_DIR, "TrustMark_denoising_attack")
ATTACKED_DIR = os.path.join(OUTPUT_DIR, "attacked_images")
EXCEL_PATH = os.path.join(OUTPUT_DIR, "TrustMark_denoising_results.xls")

PRIVATE_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUBLIC_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")

WATERMARK_TEXT = "ASys Encryption"
H_VALUES = [5, 10, 15]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Must match run_trustmark_export.py. Leave as None to derive it from RSA-2048.
# With the same keys, text and derivation rule, the result is deterministic.
EXPECTED_PAYLOAD_OVERRIDE = None

# TrustMark settings must match the export script.
TRUSTMARK_MODEL_TYPE = "Q"
TRUSTMARK_USE_ECC = True
TRUSTMARK_VERBOSE = True


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ATTACKED_DIR, exist_ok=True)


def load_and_validate_rsa():
    if not os.path.isfile(PRIVATE_KEY_PATH):
        raise FileNotFoundError("Missing private key: " + PRIVATE_KEY_PATH)
    if not os.path.isfile(PUBLIC_KEY_PATH):
        raise FileNotFoundError("Missing public key: " + PUBLIC_KEY_PATH)

    with open(PRIVATE_KEY_PATH, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    with open(PUBLIC_KEY_PATH, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    if private_key.key_size != 2048 or public_key.key_size != 2048:
        raise ValueError("This experiment requires RSA-2048 keys")
    if private_key.public_key().public_numbers() != public_key.public_numbers():
        raise ValueError("private_key.pem and public_key.pem do not match")

    signature = private_key.sign(
        WATERMARK_TEXT.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    public_key.verify(
        signature,
        WATERMARK_TEXT.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return private_key, public_key, signature


def derive_payload(signature):
    """Create an 8-character TrustMark payload from the external RSA signature."""
    digest = hashlib.sha256(signature).digest()
    return base64.b32encode(digest).decode("ascii").rstrip("=")[:8]


def get_image_files(folder):
    paths = []
    if not os.path.isdir(folder):
        return paths
    for root, _, files in os.walk(folder):
        for filename in sorted(files):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def read_bgr(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def denoising_attack(image, h_value):
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=h_value, hColor=h_value,
        templateWindowSize=7, searchWindowSize=21,
    )


def calculate_psnr(reference, test):
    mse = float(np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def calculate_ssim(reference, test):
    reference = reference.astype(np.float64)
    test = test.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    channel_scores = []
    for channel in range(3):
        x = reference[:, :, channel]
        y = test[:, :, channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        channel_scores.append(float(np.mean(numerator / (denominator + 1e-12))))
    return float(np.mean(channel_scores))


def normalize_decode_result(raw_result):
    """Support the common TrustMark tuple and dictionary return formats."""
    decoded = ""
    present = False
    schema = ""

    if isinstance(raw_result, dict):
        decoded = raw_result.get("secret", raw_result.get("watermark", raw_result.get("payload", "")))
        present = raw_result.get("present", raw_result.get("detected", bool(decoded)))
        schema = raw_result.get("schema", "")
    elif isinstance(raw_result, (tuple, list)):
        if len(raw_result) > 0:
            decoded = raw_result[0]
        if len(raw_result) > 1:
            present = bool(raw_result[1])
        else:
            present = bool(decoded)
        if len(raw_result) > 2:
            schema = raw_result[2]
    else:
        decoded = raw_result
        present = bool(decoded)

    if isinstance(decoded, bytes):
        decoded = decoded.decode("utf-8", errors="replace")
    decoded = "" if decoded is None else str(decoded).strip()
    schema = "" if schema is None else str(schema)
    return decoded, bool(present), schema


def decode_trustmark(decoder, attacked_bgr):
    rgb = cv2.cvtColor(attacked_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return normalize_decode_result(decoder.decode(pil_image))


def summarize(rows):
    summaries = []
    for h_value in H_VALUES:
        subset = [r for r in rows if r["h"] == h_value and r["status"] == "OK"]
        if not subset:
            continue
        n = len(subset)
        detected = sum(int(r["detected"]) for r in subset)
        exact = sum(int(r["exact"]) for r in subset)
        summaries.append({
            "h": h_value,
            "n": n,
            "detected": detected,
            "detection_rate": detected / n,
            "exact": exact,
            "exact_rate": exact / n,
            "psnr_mean": float(np.mean([r["psnr"] for r in subset])),
            "psnr_std": float(np.std([r["psnr"] for r in subset])),
            "ssim_mean": float(np.mean([r["ssim"] for r in subset])),
            "ssim_std": float(np.std([r["ssim"] for r in subset])),
            "attack_time_mean": float(np.mean([r["attack_time"] for r in subset])),
            "decode_time_mean": float(np.mean([r["decode_time"] for r in subset])),
        })
    return summaries


def write_excel(rows, summaries, expected_payload):
    workbook = xlwt.Workbook()
    detail = workbook.add_sheet("Per_Image")
    headers = [
        "Image", "h", "Status", "Detected", "Decoded payload", "Expected payload",
        "Exact payload", "RSA-bound payload verified", "Schema", "PSNR (dB)", "SSIM",
        "Attack time (s)", "Decode time (s)", "Source path", "Attacked path", "Note",
    ]
    for c, value in enumerate(headers):
        detail.write(0, c, value)
    for r_index, row in enumerate(rows, 1):
        values = [
            row.get("image", ""), row.get("h", ""), row.get("status", ""),
            str(row.get("detected", "")), row.get("decoded", ""), expected_payload,
            str(row.get("exact", "")), str(row.get("rsa_bound_verified", "")),
            row.get("schema", ""), row.get("psnr", ""), row.get("ssim", ""),
            row.get("attack_time", ""), row.get("decode_time", ""),
            row.get("source_path", ""), row.get("attacked_path", ""), row.get("note", ""),
        ]
        for c, value in enumerate(values):
            detail.write(r_index, c, value)

    summary = workbook.add_sheet("Summary")
    summary_headers = [
        "h", "N", "Detected", "Detection rate", "Exact payload", "Exact payload rate",
        "PSNR mean", "PSNR std", "SSIM mean", "SSIM std",
        "Mean attack time (s)", "Mean decode time (s)",
    ]
    for c, value in enumerate(summary_headers):
        summary.write(0, c, value)
    for r_index, item in enumerate(summaries, 1):
        values = [
            item["h"], item["n"], item["detected"], item["detection_rate"],
            item["exact"], item["exact_rate"], item["psnr_mean"], item["psnr_std"],
            item["ssim_mean"], item["ssim_std"], item["attack_time_mean"],
            item["decode_time_mean"],
        ]
        for c, value in enumerate(values):
            summary.write(r_index, c, value)

    definitions = workbook.add_sheet("Definitions")
    notes = [
        "Input folder: TrustMark_encoded (subfolders are scanned recursively).",
        "Attack: OpenCV fastNlMeansDenoisingColored, hColor=h, templateWindowSize=7, searchWindowSize=21.",
        "Detected: the TrustMark decoder reports that a watermark is present.",
        "Exact payload: decoded text exactly equals the expected 8-character RSA-derived payload.",
        "RSA-bound payload verified: exact payload match after the external RSA-2048 key pair and signature have been validated.",
        "PSNR and SSIM compare the attacked image with the encoded TrustMark image.",
        "Expected payload: " + expected_payload,
    ]
    for r_index, value in enumerate(notes):
        definitions.write(r_index, 0, value)
    workbook.save(EXCEL_PATH)


def main():
    ensure_dirs()
    _, _, signature = load_and_validate_rsa()
    expected_payload = EXPECTED_PAYLOAD_OVERRIDE or derive_payload(signature)

    print("=" * 76)
    print("TRUSTMARK + RSA-2048 DENOISING ATTACK TEST")
    print("=" * 76)
    print("Input folder     :", TRUSTMARK_DIR)
    print("Expected payload :", expected_payload)
    print("Denoising h      :", H_VALUES)

    image_paths = get_image_files(TRUSTMARK_DIR)
    if not image_paths:
        print("No images found in:", TRUSTMARK_DIR)
        return
    print("Images found     :", len(image_paths))
    print("Initializing TrustMark decoder...")
    decoder = TrustMark(
        verbose=TRUSTMARK_VERBOSE,
        model_type=TRUSTMARK_MODEL_TYPE,
        use_ECC=TRUSTMARK_USE_ECC,
    )

    rows = []
    total = len(image_paths) * len(H_VALUES)
    current = 0
    total_start = time.perf_counter()

    for source_path in image_paths:
        relative = os.path.relpath(source_path, TRUSTMARK_DIR)
        relative_stem = os.path.splitext(relative)[0]
        image_id = relative.replace(os.sep, "/")
        source = read_bgr(source_path)

        for h_value in H_VALUES:
            current += 1
            print("[%d/%d] %s | h=%d" % (current, total, image_id, h_value))
            row = {"image": image_id, "h": h_value, "source_path": source_path}
            if source is None:
                row.update({"status": "FAILED", "note": "Cannot read source image"})
                rows.append(row)
                continue
            try:
                started = time.perf_counter()
                attacked = denoising_attack(source, h_value)
                attack_time = time.perf_counter() - started

                attacked_path = os.path.join(ATTACKED_DIR, "h_%d" % h_value, relative_stem + ".png")
                os.makedirs(os.path.dirname(attacked_path), exist_ok=True)
                if not cv2.imwrite(attacked_path, attacked):
                    raise IOError("Failed to save attacked image")

                started = time.perf_counter()
                decoded, detected, schema = decode_trustmark(decoder, attacked)
                decode_time = time.perf_counter() - started
                exact = decoded == expected_payload

                row.update({
                    "status": "OK", "detected": detected, "decoded": decoded,
                    "exact": exact, "rsa_bound_verified": exact, "schema": schema,
                    "psnr": calculate_psnr(source, attacked),
                    "ssim": calculate_ssim(source, attacked),
                    "attack_time": attack_time, "decode_time": decode_time,
                    "attacked_path": attacked_path, "note": "",
                })
            except Exception as error:
                row.update({"status": "FAILED", "note": repr(error)})
                print("  ERROR:", repr(error))
            rows.append(row)

    summaries = summarize(rows)
    write_excel(rows, summaries, expected_payload)

    print("\n" + "=" * 76)
    print("TRUSTMARK DENOISING SUMMARY")
    print("=" * 76)
    for item in summaries:
        print(
            "h=%-2d N=%-4d Detect=%.4f Exact=%.4f PSNR=%.3f+/-%.3f SSIM=%.4f+/-%.4f"
            % (item["h"], item["n"], item["detection_rate"], item["exact_rate"],
               item["psnr_mean"], item["psnr_std"], item["ssim_mean"], item["ssim_std"])
        )
    failed = sum(1 for row in rows if row["status"] != "OK")
    elapsed = time.perf_counter() - total_start
    print("\nCompleted operations:", len(rows) - failed)
    print("Failed operations   :", failed)
    print("Total runtime       : %.2f seconds (%.2f minutes)" % (elapsed, elapsed / 60.0))
    print("Excel result        :", EXCEL_PATH)
    print("Attacked images     :", ATTACKED_DIR)


if __name__ == "__main__":
    main()
