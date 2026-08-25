import os
import csv
import xlwt
import cv2
import numpy as np
import pywt

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization


def pick_base_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)

    if (
        os.path.exists(os.path.join(here, "private_key.pem"))
        or os.path.isdir(os.path.join(here, "Encoded_jpeg"))
    ):
        return here

    if (
        os.path.exists(os.path.join(parent, "private_key.pem"))
        or os.path.isdir(os.path.join(parent, "Encoded_jpeg"))
    ):
        return parent

    return parent


BASE_DIR = pick_base_dir()

JPEG_DIR = os.path.join(BASE_DIR, "Encoded_jpeg")
COMPARISON_JPEG_BASE_DIR = os.path.join(
    BASE_DIR,
    "Comparison_jpeg_result"
)

PRIV_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUB_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")

WATERMARK_TEXT = "ASys Encryption"

DCT_Q = 8.0
DWT_Q = 8.0
DCT_BITS_PER_BLOCK = 4


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as file:
        return serialization.load_pem_private_key(
            file.read(),
            password=None
        )


def load_public_key():
    with open(PUB_KEY_PATH, "rb") as file:
        return serialization.load_pem_public_key(file.read())


def sign_message(private_key, message):
    return private_key.sign(
        message.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256()
    )


def verify_signature(public_key, message, signature):
    if not signature:
        return False

    try:
        public_key.verify(
            signature,
            message.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return True

    except Exception:
        return False


def bytes_to_bits(data):
    return "".join(f"{byte:08b}" for byte in data)


def bits_to_bytes(bits):
    usable_length = len(bits) - len(bits) % 8

    if usable_length <= 0:
        return b""

    bits = bits[:usable_length]

    return bytes(
        int(bits[i:i + 8], 2)
        for i in range(0, len(bits), 8)
    )


def read_cv(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image is None:
        return None

    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    elif image.shape[2] == 4:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

    return image


def decode_lsb_safe(image):
    flat = image.reshape(-1, 3)

    if flat.shape[0] < 32:
        return b""

    header_values = flat[:32, 0]
    header_bits = "".join(
        str(int(value) & 1)
        for value in header_values
    )

    try:
        length = int(header_bits, 2)

    except ValueError:
        return b""

    maximum_length = flat.shape[0] - 32

    if length <= 0 or length > maximum_length:
        return b""

    extracted_values = flat[32:32 + length, 0]

    bits = "".join(
        str(int(value) & 1)
        for value in extracted_values
    )

    return bits_to_bytes(bits)


def decode_dct(
    image,
    bits_len,
    Q=DCT_Q,
    bits_per_block=DCT_BITS_PER_BLOCK
):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    blocks_y = height // 8
    blocks_x = width // 8

    positions = [
        (4, 4),
        (3, 3),
        (2, 2),
        (5, 5)
    ]

    bits = np.empty(bits_len, dtype=np.uint8)
    bit_index = 0

    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            y0 = block_y * 8
            x0 = block_x * 8

            block = np.float32(
                gray[y0:y0 + 8, x0:x0 + 8]
            )

            dct_block = cv2.dct(block)

            for row, column in positions[:bits_per_block]:
                quantized_value = int(
                    np.rint(dct_block[row, column] / Q)
                )

                bits[bit_index] = quantized_value & 1
                bit_index += 1

                if bit_index >= bits_len:
                    return np.packbits(bits).tobytes()

    usable_length = bit_index - bit_index % 8

    if usable_length <= 0:
        return b""

    return np.packbits(
        bits[:usable_length]
    ).tobytes()


def decode_dwt(image, bits_len, Q=DWT_Q):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, (LH, _, _) = pywt.dwt2(
        np.float32(gray),
        "haar"
    )

    flat = LH.ravel()

    length = min(bits_len, flat.size)
    length -= length % 8

    if length <= 0:
        return b""

    quantized_values = np.rint(
        flat[:length] / Q
    ).astype(np.int64)

    extracted_bits = (
        quantized_values & 1
    ).astype(np.uint8)

    return np.packbits(
        extracted_bits
    ).tobytes()


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
    os.makedirs(
        COMPARISON_JPEG_BASE_DIR,
        exist_ok=True
    )

    print("Base folder:")
    print(BASE_DIR)

    print("\nJPEG input folder:")
    print(JPEG_DIR)

    if not os.path.exists(PRIV_KEY_PATH):
        print("\nPrivate key not found:")
        print(PRIV_KEY_PATH)
        return

    if not os.path.exists(PUB_KEY_PATH):
        print("\nPublic key not found:")
        print(PUB_KEY_PATH)
        return

    try:
        private_key = load_private_key()
        public_key = load_public_key()

    except Exception as error:
        print("\nFailed to load RSA keys:")
        print(error)
        return

    reference_signature = sign_message(
        private_key,
        WATERMARK_TEXT
    )

    signature_byte_length = len(reference_signature)
    signature_bit_length = signature_byte_length * 8

    print("\nRSA signature length:")
    print(signature_byte_length, "bytes")
    print(signature_bit_length, "bits")

    if signature_bit_length == 2048:
        print("RSA key size: 2048 bits")
    else:
        print(
            "Warning: expected an RSA-2048 signature, "
            f"but found {signature_bit_length} bits."
        )

    quality_dirs = list_quality_dirs()

    if not quality_dirs:
        print("\nNo JPEG quality folders found in:")
        print(JPEG_DIR)
        return

    per_image_csv = os.path.join(
        COMPARISON_JPEG_BASE_DIR,
        "per_image_jpeg.csv"
    )

    summary_csv = os.path.join(
        COMPARISON_JPEG_BASE_DIR,
        "summary_jpeg.csv"
    )

    summary_xls = os.path.join(
        COMPARISON_JPEG_BASE_DIR,
        "summary_jpeg.xls"
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

        print()
        print(
            f"[{quality_index}/{total_quality_count}] "
            f"Processing Q{quality}"
        )
        print("Number of image folders:", len(folders))

        total = len(folders)

        lsb_ok = 0
        dct_ok = 0
        dwt_ok = 0

        lsb_missing = 0
        dct_missing = 0
        dwt_missing = 0

        for image_index, folder_name in enumerate(
            folders,
            start=1
        ):
            if (
                image_index == 1
                or image_index % 100 == 0
                or image_index == total
            ):
                print(
                    f"Q{quality}: "
                    f"{image_index}/{total}"
                )

            folder_path = os.path.join(
                quality_path,
                folder_name
            )

            lsb_valid = False
            dct_valid = False
            dwt_valid = False

            lsb_status = "not_valid"
            dct_status = "not_valid"
            dwt_status = "not_valid"

            lsb_path = os.path.join(
                folder_path,
                "LSB.jpg"
            )

            if os.path.exists(lsb_path):
                image = read_cv(lsb_path)

                if image is not None:
                    decoded_signature = decode_lsb_safe(
                        image
                    )

                    lsb_valid = verify_signature(
                        public_key,
                        WATERMARK_TEXT,
                        decoded_signature
                    )

                    if lsb_valid:
                        lsb_status = "valid"
                    else:
                        lsb_status = "not_valid"

                else:
                    lsb_status = "read_failed"
                    lsb_missing += 1

            else:
                lsb_status = "missing"
                lsb_missing += 1

            dct_path = os.path.join(
                folder_path,
                "DCT.jpg"
            )

            if os.path.exists(dct_path):
                image = read_cv(dct_path)

                if image is not None:
                    decoded_signature = decode_dct(
                        image,
                        signature_bit_length
                    )

                    dct_valid = verify_signature(
                        public_key,
                        WATERMARK_TEXT,
                        decoded_signature
                    )

                    if dct_valid:
                        dct_status = "valid"
                    else:
                        dct_status = "not_valid"

                else:
                    dct_status = "read_failed"
                    dct_missing += 1

            else:
                dct_status = "missing"
                dct_missing += 1

            dwt_path = os.path.join(
                folder_path,
                "DWT.jpg"
            )

            if os.path.exists(dwt_path):
                image = read_cv(dwt_path)

                if image is not None:
                    decoded_signature = decode_dwt(
                        image,
                        signature_bit_length
                    )

                    dwt_valid = verify_signature(
                        public_key,
                        WATERMARK_TEXT,
                        decoded_signature
                    )

                    if dwt_valid:
                        dwt_status = "valid"
                    else:
                        dwt_status = "not_valid"

                else:
                    dwt_status = "read_failed"
                    dwt_missing += 1

            else:
                dwt_status = "missing"
                dwt_missing += 1

            if lsb_valid:
                lsb_ok += 1

            if dct_valid:
                dct_ok += 1

            if dwt_valid:
                dwt_ok += 1

            per_image_rows.append([
                quality,
                folder_name,
                lsb_valid,
                lsb_status,
                dct_valid,
                dct_status,
                dwt_valid,
                dwt_status
            ])

        lsb_rate = lsb_ok / total if total else 0.0
        dct_rate = dct_ok / total if total else 0.0
        dwt_rate = dwt_ok / total if total else 0.0

        summary_rows.append([
            quality,
            total,
            lsb_ok,
            lsb_rate,
            lsb_missing,
            dct_ok,
            dct_rate,
            dct_missing,
            dwt_ok,
            dwt_rate,
            dwt_missing
        ])

        print(f"Q{quality} completed:")
        print(
            f"LSB: {lsb_ok}/{total} "
            f"({lsb_rate:.4%})"
        )
        print(
            f"DCT: {dct_ok}/{total} "
            f"({dct_rate:.4%})"
        )
        print(
            f"DWT: {dwt_ok}/{total} "
            f"({dwt_rate:.4%})"
        )

    per_image_headers = [
        "jpeg_quality",
        "image_folder",
        "lsb_valid",
        "lsb_status",
        "dct_valid",
        "dct_status",
        "dwt_valid",
        "dwt_status"
    ]

    summary_headers = [
        "jpeg_quality",
        "total_images",
        "lsb_valid_count",
        "lsb_valid_rate",
        "lsb_missing_count",
        "dct_valid_count",
        "dct_valid_rate",
        "dct_missing_count",
        "dwt_valid_count",
        "dwt_valid_rate",
        "dwt_missing_count"
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

    for column, header in enumerate(summary_headers):
        worksheet.write(0, column, header)

    percentage_style = xlwt.easyxf(
        num_format_str="0.0000%"
    )

    for row_index, row in enumerate(
        summary_rows,
        start=1
    ):
        for column_index, value in enumerate(row):
            if column_index in [3, 6, 9]:
                worksheet.write(
                    row_index,
                    column_index,
                    value,
                    percentage_style
                )
            else:
                worksheet.write(
                    row_index,
                    column_index,
                    value
                )

    workbook.save(summary_xls)

    print("\nAll JPEG images have been processed.")
    print("\nSaved:")
    print(per_image_csv)
    print(summary_csv)
    print(summary_xls)


if __name__ == "__main__":
    main()