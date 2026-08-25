import os
import csv
import xlwt
import cv2
import numpy as np
import hashlib
import pywt

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization


def pick_base_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)

    if (
        os.path.isdir(os.path.join(here, "Spread_jpeg"))
        or os.path.exists(os.path.join(here, "private_key.pem"))
    ):
        return here

    if (
        os.path.isdir(os.path.join(parent, "Spread_jpeg"))
        or os.path.exists(os.path.join(parent, "private_key.pem"))
    ):
        return parent

    return parent


BASE_DIR = pick_base_dir()

JPEG_DIR = os.path.join(
    BASE_DIR,
    "Spread_jpeg"
)

COMPARISON_DIR = os.path.join(
    BASE_DIR,
    "Comparison_dwtss_jpeg_result"
)

PRIV_KEY_PATH = os.path.join(
    BASE_DIR,
    "private_key.pem"
)

WATERMARK_TEXT = "ASys Encryption"

BER_THRESHOLD = 0.25

SEED_BASE = int.from_bytes(
    hashlib.sha256(b"KEY").digest()[:4],
    "big"
)


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as file:
        return serialization.load_pem_private_key(
            file.read(),
            password=None
        )


def sign_message(private_key, message):
    return private_key.sign(
        message.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256()
    )


def bytes_to_bits(data):
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8)
    )


def get_watermark_bits(private_key):
    signature = sign_message(
        private_key,
        WATERMARK_TEXT
    )

    digest = hashlib.sha256(signature).digest()

    return bytes_to_bits(digest[:4])


def detect_spread_asym(
    image_bgr,
    reference_bits
):
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    LL, (_, _, _) = pywt.dwt2(
        gray,
        "haar"
    )

    flat = LL.ravel()

    coefficient_count = flat.size
    number_of_bits = reference_bits.size

    if number_of_bits == 0:
        return False, 1.0

    chunk_size = coefficient_count // number_of_bits

    if chunk_size <= 0:
        return False, 1.0

    received_bits = np.zeros(
        number_of_bits,
        dtype=np.uint8
    )

    for bit_index in range(number_of_bits):
        start = bit_index * chunk_size
        end = min(
            start + chunk_size,
            coefficient_count
        )

        segment = flat[start:end]

        if segment.size == 0:
            received_bits[bit_index] = 0
            continue

        random_generator = np.random.RandomState(
            SEED_BASE + bit_index
        )

        pseudo_noise = random_generator.choice(
            [-1.0, 1.0],
            size=segment.size
        ).astype(np.float32)

        correlation = float(
            np.dot(segment, pseudo_noise)
        )

        received_bits[bit_index] = (
            1 if correlation >= 0 else 0
        )

    error_count = int(
        np.count_nonzero(
            received_bits != reference_bits
        )
    )

    ber = error_count / number_of_bits
    verified = ber < BER_THRESHOLD

    return bool(verified), float(ber)


def list_quality_dirs():
    if not os.path.isdir(JPEG_DIR):
        return []

    quality_dirs = []

    for folder_name in os.listdir(JPEG_DIR):
        folder_path = os.path.join(
            JPEG_DIR,
            folder_name
        )

        if (
            os.path.isdir(folder_path)
            and folder_name.startswith("Q")
        ):
            try:
                quality = int(folder_name[1:])

                quality_dirs.append(
                    (quality, folder_path)
                )

            except ValueError:
                continue

    quality_dirs.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return quality_dirs


def main():
    print("Base folder:")
    print(BASE_DIR)

    print("\nDWT-SS JPEG input folder:")
    print(JPEG_DIR)

    if not os.path.exists(PRIV_KEY_PATH):
        print("\nPrivate key not found:")
        print(PRIV_KEY_PATH)
        return

    try:
        private_key = load_private_key()

    except Exception as error:
        print("\nFailed to load private key:")
        print(error)
        return

    reference_bits = get_watermark_bits(
        private_key
    )

    print("\nReference watermark length:")
    print(reference_bits.size, "bits")

    print("BER threshold:")
    print(BER_THRESHOLD)

    os.makedirs(
        COMPARISON_DIR,
        exist_ok=True
    )

    quality_dirs = list_quality_dirs()

    if not quality_dirs:
        print("\nNo JPEG folders found in:")
        print(JPEG_DIR)
        return

    per_image_csv = os.path.join(
        COMPARISON_DIR,
        "per_image_dwtss_jpeg.csv"
    )

    summary_csv = os.path.join(
        COMPARISON_DIR,
        "summary_dwtss_jpeg.csv"
    )

    summary_xls = os.path.join(
        COMPARISON_DIR,
        "summary_dwtss_jpeg.xls"
    )

    per_image_rows = []
    summary_rows = []

    total_quality_count = len(quality_dirs)

    for quality_index, (quality, quality_path) in enumerate(
        quality_dirs,
        start=1
    ):
        folders = sorted([
            folder_name
            for folder_name in os.listdir(quality_path)
            if os.path.isdir(
                os.path.join(
                    quality_path,
                    folder_name
                )
            )
        ])

        folder_count = len(folders)

        print()
        print(
            f"[{quality_index}/{total_quality_count}] "
            f"Processing Q{quality}"
        )
        print(
            "Number of image folders:",
            folder_count
        )

        total = 0
        verified_count = 0
        ber_sum = 0.0
        missing_count = 0

        for image_index, folder_name in enumerate(
            folders,
            start=1
        ):
            if (
                image_index == 1
                or image_index % 100 == 0
                or image_index == folder_count
            ):
                print(
                    f"Q{quality}: "
                    f"{image_index}/{folder_count}"
                )

            image_path = os.path.join(
                quality_path,
                folder_name,
                "spread.jpg"
            )

            if not os.path.exists(image_path):
                missing_count += 1

                per_image_rows.append([
                    quality,
                    folder_name,
                    False,
                    "",
                    "missing"
                ])

                continue

            image = cv2.imread(
                image_path,
                cv2.IMREAD_COLOR
            )

            if image is None:
                missing_count += 1

                per_image_rows.append([
                    quality,
                    folder_name,
                    False,
                    "",
                    "read_failed"
                ])

                continue

            total += 1

            verified, ber = detect_spread_asym(
                image,
                reference_bits
            )

            if verified:
                verified_count += 1

            ber_sum += ber

            per_image_rows.append([
                quality,
                folder_name,
                verified,
                ber,
                "processed"
            ])

        verified_rate = (
            verified_count / total
            if total
            else 0.0
        )

        mean_ber = (
            ber_sum / total
            if total
            else 0.0
        )

        summary_rows.append([
            quality,
            total,
            verified_count,
            verified_rate,
            mean_ber,
            missing_count
        ])

        print(f"Q{quality} completed:")
        print(
            f"Verified: {verified_count}/{total} "
            f"({verified_rate:.4%})"
        )
        print(f"Mean BER: {mean_ber:.6f}")
        print(f"Missing/read failed: {missing_count}")

    per_image_headers = [
        "jpeg_quality",
        "image_folder",
        "verified",
        "ber",
        "status"
    ]

    summary_headers = [
        "jpeg_quality",
        "total_images",
        "verified_count",
        "verified_rate",
        "mean_ber",
        "missing_count"
    ]

    with open(
        per_image_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(per_image_headers)
        writer.writerows(per_image_rows)

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(summary_headers)
        writer.writerows(summary_rows)

    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet("summary")

    for column_index, header in enumerate(
        summary_headers
    ):
        worksheet.write(
            0,
            column_index,
            header
        )

    percentage_style = xlwt.easyxf(
        num_format_str="0.0000%"
    )

    decimal_style = xlwt.easyxf(
        num_format_str="0.000000"
    )

    for row_index, row in enumerate(
        summary_rows,
        start=1
    ):
        for column_index, value in enumerate(row):
            if column_index == 3:
                worksheet.write(
                    row_index,
                    column_index,
                    value,
                    percentage_style
                )

            elif column_index == 4:
                worksheet.write(
                    row_index,
                    column_index,
                    value,
                    decimal_style
                )

            else:
                worksheet.write(
                    row_index,
                    column_index,
                    value
                )

    workbook.save(summary_xls)

    print(
        "\nAll DWT-SS JPEG images "
        "have been processed."
    )

    print("\nSaved:")
    print(per_image_csv)
    print(summary_csv)
    print(summary_xls)


if __name__ == "__main__":
    main()