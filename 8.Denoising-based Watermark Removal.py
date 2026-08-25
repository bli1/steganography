import os
import time
import math
import hashlib
import binascii

import cv2
import numpy as np
import pywt
import xlwt

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORIGINAL_DIR = os.path.join(BASE_DIR, "Original_image")
CLASSIC_DIR = os.path.join(BASE_DIR, "Encoded_image")
SPATIAL_SS_DIR = os.path.join(BASE_DIR, "Spatial_encoded")
DWT_SS_DIR = os.path.join(BASE_DIR, "Spread_encoded")

OUTPUT_DIR = os.path.join(BASE_DIR, "All_denoising_attack_results")
ATTACKED_DIR = os.path.join(OUTPUT_DIR, "attacked_images")
DETAIL_DIR = os.path.join(OUTPUT_DIR, "decoded_details")
EXCEL_PATH = os.path.join(OUTPUT_DIR, "all_methods_denoising_results.xls")

PRIV_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUB_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")

WATERMARK_TEXT = "ASys Encryption"
H_VALUES = [5, 10, 15]
DETECTION_BER_THRESHOLD = 0.25
ALPHA = 20.0

DCT_Q = 8.0
DCT_BITS_PER_BLOCK = 4
DWT_Q = 8.0

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ATTACKED_DIR, exist_ok=True)
    os.makedirs(DETAIL_DIR, exist_ok=True)


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public_key():
    with open(PUB_KEY_PATH, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def validate_keys():
    if not os.path.isfile(PRIV_KEY_PATH):
        raise FileNotFoundError("Missing private key: " + PRIV_KEY_PATH)
    if not os.path.isfile(PUB_KEY_PATH):
        raise FileNotFoundError("Missing public key: " + PUB_KEY_PATH)

    private_key = load_private_key()
    public_key = load_public_key()

    private_bits = private_key.key_size
    public_bits = public_key.key_size

    if private_bits != public_bits:
        raise ValueError("Private and public RSA key sizes do not match")

    if private_key.public_key().public_numbers() != public_key.public_numbers():
        raise ValueError("private_key.pem and public_key.pem are not a matching pair")

    print("RSA key size:", private_bits)
    if private_bits < 2048:
        print("WARNING: RSA key size is below 2048 bits.")

    return private_key, public_key


def sign_message(private_key):
    return private_key.sign(
        WATERMARK_TEXT.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def verify_signature(public_key, signature):
    try:
        public_key.verify(
            signature,
            WATERMARK_TEXT.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def bytes_to_bit_array(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bit_array_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    usable = (bits.size // 8) * 8
    if usable == 0:
        return b""
    return np.packbits(bits[:usable]).tobytes()


def calculate_ber(received_bits, reference_bits):
    received_bits = np.asarray(received_bits, dtype=np.uint8)
    reference_bits = np.asarray(reference_bits, dtype=np.uint8)

    if reference_bits.size == 0:
        return 1.0

    if received_bits.size < reference_bits.size:
        padded = np.zeros(reference_bits.size, dtype=np.uint8)
        padded[:received_bits.size] = received_bits
        received_bits = padded
    else:
        received_bits = received_bits[:reference_bits.size]

    return float(np.mean(received_bits != reference_bits))


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def denoising_attack(image, h):
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=h,
        hColor=h,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def calculate_psnr(reference, test):
    if reference.shape != test.shape:
        test = cv2.resize(test, (reference.shape[1], reference.shape[0]))
    mse = float(np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def calculate_ssim(reference, test):
    if reference.shape != test.shape:
        test = cv2.resize(test, (reference.shape[1], reference.shape[0]))

    reference = reference.astype(np.float64)
    test = test.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    values = []

    for channel in range(reference.shape[2]):
        x = reference[:, :, channel]
        y = test[:, :, channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        values.append(float(np.mean(numerator / (denominator + 1e-12))))

    return float(np.mean(values))


def extract_lsb_bits(image, signature_bits_length):
    flat = image.reshape(-1, 3)
    start = 32
    available = max(0, flat.shape[0] - start)
    count = min(signature_bits_length, available)
    bits = np.zeros(signature_bits_length, dtype=np.uint8)
    if count:
        bits[:count] = flat[start:start + count, 0] & 1
    return bits


def extract_dct_bits(image, bits_length, q_value=DCT_Q, bits_per_block=DCT_BITS_PER_BLOCK):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blocks_y = gray.shape[0] // 8
    blocks_x = gray.shape[1] // 8
    positions = [(4, 4), (3, 3), (2, 2), (5, 5)]
    bits = []

    for by in range(blocks_y):
        for bx in range(blocks_x):
            if len(bits) >= bits_length:
                break
            block = np.float32(gray[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8])
            dct_block = cv2.dct(block)
            for index in range(bits_per_block):
                if len(bits) >= bits_length:
                    break
                row, col = positions[index]
                quantized = int(np.round(dct_block[row, col] / q_value))
                bits.append(quantized & 1)
        if len(bits) >= bits_length:
            break

    result = np.zeros(bits_length, dtype=np.uint8)
    result[:len(bits)] = bits
    return result


def extract_dwt_bits(image, bits_length, q_value=DWT_Q):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    _, (lh, _, _) = pywt.dwt2(gray, "haar")
    flat = lh.reshape(-1)
    count = min(bits_length, flat.size)
    result = np.zeros(bits_length, dtype=np.uint8)
    for index in range(count):
        quantized = int(np.round(flat[index] / q_value))
        result[index] = quantized & 1
    return result


def get_fingerprint_bits(reference_signature):
    digest = hashlib.sha256(reference_signature).digest()
    return bytes_to_bit_array(digest[:4])


def extract_spatial_ss_bits(image, number_of_bits=32):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    flat = gray.reshape(-1)
    chunk = flat.size // number_of_bits
    seed_base = int.from_bytes(hashlib.sha256(b"SPATIAL_KEY").digest()[:4], "big")
    received = np.zeros(number_of_bits, dtype=np.uint8)

    for index in range(number_of_bits):
        start = index * chunk
        end = min(start + chunk, flat.size)
        segment = flat[start:end]
        rng = np.random.RandomState(seed_base + index)
        pn = rng.choice([-1, 1], size=segment.size).astype(np.float32)
        received[index] = 1 if float(np.sum(segment * pn)) >= 0 else 0

    return received


def extract_dwt_ss_bits(image, number_of_bits=32):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ll, _ = pywt.dwt2(gray, "haar")
    flat = ll.reshape(-1)
    chunk = flat.size // number_of_bits
    seed_base = int.from_bytes(hashlib.sha256(b"KEY").digest()[:4], "big")
    received = np.zeros(number_of_bits, dtype=np.uint8)

    for index in range(number_of_bits):
        start = index * chunk
        end = min(start + chunk, flat.size)
        segment = flat[start:end]
        rng = np.random.RandomState(seed_base + index)
        pn = rng.choice([-1, 1], size=segment.size).astype(np.float32)
        received[index] = 1 if float(np.sum(segment * pn)) >= 0 else 0

    return received


def build_original_index():
    result = {}
    if not os.path.isdir(ORIGINAL_DIR):
        return result
    for root, _, files in os.walk(ORIGINAL_DIR):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                stem = os.path.splitext(filename)[0].lower()
                result.setdefault(stem, os.path.join(root, filename))
    return result


def expected_watermarked_paths():
    jobs = []

    if os.path.isdir(CLASSIC_DIR):
        for folder in sorted(os.listdir(CLASSIC_DIR)):
            folder_path = os.path.join(CLASSIC_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            for method, filename in (("LSB", "LSB.png"), ("DCT", "DCT.png"), ("DWT", "DWT.png")):
                path = os.path.join(folder_path, filename)
                if os.path.isfile(path):
                    jobs.append((method, folder, path))

    if os.path.isdir(SPATIAL_SS_DIR):
        for folder in sorted(os.listdir(SPATIAL_SS_DIR)):
            folder_path = os.path.join(SPATIAL_SS_DIR, folder)
            path = os.path.join(folder_path, "spatial_spread.png")
            if os.path.isfile(path):
                jobs.append(("Spatial SS", folder, path))

    if os.path.isdir(DWT_SS_DIR):
        for folder in sorted(os.listdir(DWT_SS_DIR)):
            folder_path = os.path.join(DWT_SS_DIR, folder)
            path = os.path.join(folder_path, "spread.png")
            if os.path.isfile(path):
                jobs.append(("DWT-SS", folder, path))

    return jobs


def decode_method(method, attacked_image, reference_signature, reference_signature_bits, fingerprint_bits, public_key):
    if method == "LSB":
        received_bits = extract_lsb_bits(attacked_image, reference_signature_bits.size)
    elif method == "DCT":
        received_bits = extract_dct_bits(attacked_image, reference_signature_bits.size)
    elif method == "DWT":
        received_bits = extract_dwt_bits(attacked_image, reference_signature_bits.size)
    elif method == "Spatial SS":
        received_bits = extract_spatial_ss_bits(attacked_image, fingerprint_bits.size)
    elif method == "DWT-SS":
        received_bits = extract_dwt_ss_bits(attacked_image, fingerprint_bits.size)
    else:
        raise ValueError("Unknown method: " + method)

    if method in ("LSB", "DCT", "DWT"):
        reference_bits = reference_signature_bits
        received_bytes = bit_array_to_bytes(received_bits)
        exact = bool(np.array_equal(received_bits, reference_bits))
        authentication_verified = verify_signature(public_key, received_bytes)
        decoded_hex = binascii.hexlify(received_bytes).decode("ascii")
        verification_type = "RSA signature"
    else:
        reference_bits = fingerprint_bits
        exact = bool(np.array_equal(received_bits, reference_bits))
        authentication_verified = exact
        decoded_hex = binascii.hexlify(bit_array_to_bytes(received_bits)).decode("ascii")
        verification_type = "32-bit RSA-signature fingerprint"

    ber = calculate_ber(received_bits, reference_bits)
    detected = ber < DETECTION_BER_THRESHOLD

    return {
        "received_bits": received_bits,
        "reference_bits": reference_bits,
        "ber": ber,
        "detected": detected,
        "exact": exact,
        "authentication_verified": authentication_verified,
        "decoded_hex": decoded_hex,
        "verification_type": verification_type,
    }


def safe_method_name(method):
    return method.replace(" ", "_").replace("-", "_")


def save_decode_detail(method, image_id, h_value, result):
    method_dir = os.path.join(DETAIL_DIR, safe_method_name(method), f"h_{h_value}")
    os.makedirs(method_dir, exist_ok=True)
    path = os.path.join(method_dir, f"{image_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Method: " + method + "\n")
        f.write("Image: " + image_id + "\n")
        f.write("Denoising h: " + str(h_value) + "\n")
        f.write("Verification type: " + result["verification_type"] + "\n")
        f.write("Detected (BER < %.2f): %s\n" % (DETECTION_BER_THRESHOLD, result["detected"]))
        f.write("Exact extraction: " + str(result["exact"]) + "\n")
        f.write("Authentication/fingerprint verified: " + str(result["authentication_verified"]) + "\n")
        f.write("BER: %.8f\n" % result["ber"])
        f.write("Decoded hex: " + result["decoded_hex"] + "\n")
        f.write("Received bits: " + "".join(str(int(x)) for x in result["received_bits"]) + "\n")
        f.write("Reference bits: " + "".join(str(int(x)) for x in result["reference_bits"]) + "\n")


def summarize(rows):
    summaries = []
    methods = ["LSB", "DCT", "DWT", "Spatial SS", "DWT-SS"]

    for method in methods:
        for h_value in H_VALUES:
            subset = [row for row in rows if row["method"] == method and row["h"] == h_value and row["status"] == "OK"]
            if not subset:
                continue

            count = len(subset)
            detected = sum(int(row["detected"]) for row in subset)
            exact = sum(int(row["exact"]) for row in subset)
            authenticated = sum(int(row["authentication_verified"]) for row in subset)
            ber_values = np.asarray([row["ber"] for row in subset], dtype=np.float64)
            psnr_values = np.asarray([row["psnr"] for row in subset], dtype=np.float64)
            ssim_values = np.asarray([row["ssim"] for row in subset], dtype=np.float64)
            time_values = np.asarray([row["attack_time"] for row in subset], dtype=np.float64)

            summaries.append({
                "method": method,
                "h": h_value,
                "n": count,
                "detected": detected,
                "detection_rate": detected / count,
                "exact": exact,
                "exact_rate": exact / count,
                "authenticated": authenticated,
                "authentication_rate": authenticated / count,
                "ber_mean": float(np.mean(ber_values)),
                "ber_std": float(np.std(ber_values)),
                "psnr_mean": float(np.mean(psnr_values)),
                "psnr_std": float(np.std(psnr_values)),
                "ssim_mean": float(np.mean(ssim_values)),
                "ssim_std": float(np.std(ssim_values)),
                "time_mean": float(np.mean(time_values)),
            })

    return summaries


def write_excel(rows, summaries):
    workbook = xlwt.Workbook()

    detail_sheet = workbook.add_sheet("Per_Image")
    detail_headers = [
        "Method", "Image", "h", "Status", "Detected", "Exact extraction",
        "Authentication verified", "BER", "PSNR (dB)", "SSIM",
        "Attack time (s)", "Watermarked path", "Attacked path", "Note",
    ]
    for column, header in enumerate(detail_headers):
        detail_sheet.write(0, column, header)

    for row_number, row in enumerate(rows, start=1):
        values = [
            row["method"], row["image"], row["h"], row["status"],
            str(row.get("detected", "")), str(row.get("exact", "")),
            str(row.get("authentication_verified", "")), row.get("ber", ""),
            row.get("psnr", ""), row.get("ssim", ""), row.get("attack_time", ""),
            row.get("watermarked_path", ""), row.get("attacked_path", ""), row.get("note", ""),
        ]
        for column, value in enumerate(values):
            detail_sheet.write(row_number, column, value)

    summary_sheet = workbook.add_sheet("Summary")
    summary_headers = [
        "Method", "h", "N", "Detected", "Detection rate", "Exact",
        "Exact extraction rate", "Authenticated", "Authentication rate",
        "BER mean", "BER std", "PSNR mean", "PSNR std", "SSIM mean",
        "SSIM std", "Mean attack time (s)",
    ]
    for column, header in enumerate(summary_headers):
        summary_sheet.write(0, column, header)

    for row_number, item in enumerate(summaries, start=1):
        values = [
            item["method"], item["h"], item["n"], item["detected"], item["detection_rate"],
            item["exact"], item["exact_rate"], item["authenticated"], item["authentication_rate"],
            item["ber_mean"], item["ber_std"], item["psnr_mean"], item["psnr_std"],
            item["ssim_mean"], item["ssim_std"], item["time_mean"],
        ]
        for column, value in enumerate(values):
            summary_sheet.write(row_number, column, value)

    notes_sheet = workbook.add_sheet("Definitions")
    notes = [
        "Attack: OpenCV fastNlMeansDenoisingColored with hColor=h, templateWindowSize=7, searchWindowSize=21.",
        "Detection: BER < 0.25 for all five methods.",
        "Exact extraction: all extracted bits equal the reference bits.",
        "Authentication for LSB/DCT/DWT: extracted full signature passes RSA PKCS#1 v1.5 + SHA-256 verification.",
        "Authentication for Spatial SS/DWT-SS: extracted 32-bit SHA-256 fingerprint exactly equals the reference fingerprint.",
        "PSNR and SSIM compare the attacked image with its watermarked image, measuring distortion introduced by the attack.",
        "The denoising operation is deterministic; std values in Summary are across images, not repeated identical runs.",
        "LSB extraction uses the known RSA signature length and skips the 32-bit embedded length header, because an attacked header may be corrupted.",
    ]
    for row_number, note in enumerate(notes):
        notes_sheet.write(row_number, 0, note)

    workbook.save(EXCEL_PATH)


def main():
    ensure_output_dirs()
    private_key, public_key = validate_keys()

    reference_signature = sign_message(private_key)
    reference_signature_bits = bytes_to_bit_array(reference_signature)
    fingerprint_bits = get_fingerprint_bits(reference_signature)

    print("Signature length:", len(reference_signature), "bytes /", reference_signature_bits.size, "bits")
    print("Fingerprint length:", fingerprint_bits.size, "bits")

    jobs = expected_watermarked_paths()
    if not jobs:
        print("No encoded watermark images were found.")
        print("Expected folders:")
        print(CLASSIC_DIR)
        print(SPATIAL_SS_DIR)
        print(DWT_SS_DIR)
        return

    counts = {}
    for method, _, _ in jobs:
        counts[method] = counts.get(method, 0) + 1

    print("Images found by method:")
    for method in ["LSB", "DCT", "DWT", "Spatial SS", "DWT-SS"]:
        print("  %-10s %d" % (method + ":", counts.get(method, 0)))

    rows = []
    total_operations = len(jobs) * len(H_VALUES)
    operation_number = 0
    total_start = time.perf_counter()

    for method, image_id, watermarked_path in jobs:
        watermarked_image = read_image(watermarked_path)

        for h_value in H_VALUES:
            operation_number += 1
            print("[%d/%d] %s | %s | h=%d" % (operation_number, total_operations, method, image_id, h_value))

            base_row = {
                "method": method,
                "image": image_id,
                "h": h_value,
                "watermarked_path": watermarked_path,
            }

            if watermarked_image is None:
                base_row.update({"status": "FAILED", "note": "Cannot read watermarked image"})
                rows.append(base_row)
                continue

            try:
                start = time.perf_counter()
                attacked_image = denoising_attack(watermarked_image, h_value)
                attack_time = time.perf_counter() - start

                method_output = os.path.join(ATTACKED_DIR, safe_method_name(method), f"h_{h_value}", image_id)
                os.makedirs(method_output, exist_ok=True)
                attacked_path = os.path.join(method_output, "denoised.png")

                if not cv2.imwrite(attacked_path, attacked_image):
                    raise IOError("Failed to save attacked image")

                result = decode_method(
                    method,
                    attacked_image,
                    reference_signature,
                    reference_signature_bits,
                    fingerprint_bits,
                    public_key,
                )

                psnr = calculate_psnr(watermarked_image, attacked_image)
                ssim = calculate_ssim(watermarked_image, attacked_image)
                save_decode_detail(method, image_id, h_value, result)

                base_row.update({
                    "status": "OK",
                    "detected": result["detected"],
                    "exact": result["exact"],
                    "authentication_verified": result["authentication_verified"],
                    "ber": result["ber"],
                    "psnr": psnr,
                    "ssim": ssim,
                    "attack_time": attack_time,
                    "attacked_path": attacked_path,
                    "note": result["verification_type"],
                })
                rows.append(base_row)

            except Exception as error:
                base_row.update({"status": "FAILED", "note": repr(error)})
                rows.append(base_row)
                print("  ERROR:", repr(error))

    summaries = summarize(rows)
    write_excel(rows, summaries)
    elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 76)
    print("DENOISING ATTACK SUMMARY")
    print("=" * 76)
    for item in summaries:
        print(
            "%-10s h=%-2d N=%-4d Detect=%.4f Exact=%.4f Auth=%.4f "
            "BER=%.5f+/-%.5f PSNR=%.3f SSIM=%.4f"
            % (
                item["method"], item["h"], item["n"], item["detection_rate"],
                item["exact_rate"], item["authentication_rate"], item["ber_mean"],
                item["ber_std"], item["psnr_mean"], item["ssim_mean"],
            )
        )

    failed = sum(1 for row in rows if row["status"] != "OK")
    print("\nCompleted operations:", len(rows) - failed)
    print("Failed operations:", failed)
    print("Total runtime: %.2f seconds (%.2f minutes)" % (elapsed, elapsed / 60.0))
    print("Excel result:", EXCEL_PATH)
    print("Attacked images:", ATTACKED_DIR)
    print("Decoded details:", DETAIL_DIR)


if __name__ == "__main__":
    main()
