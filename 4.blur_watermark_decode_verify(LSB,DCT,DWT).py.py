import os
import xlwt
import cv2
import numpy as np
import binascii
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
import pywt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BLUR_DIR = os.path.join(BASE_DIR, "Encoded_blur")
DECODED_BLUR_BASE_DIR = os.path.join(BASE_DIR, "Decoded_blur_output")
COMPARISON_BLUR_BASE_DIR = os.path.join(
    BASE_DIR,
    "Comparison_blur_result"
)

PRIV_KEY_PATH = os.path.join(BASE_DIR, "private_key.pem")
PUB_KEY_PATH = os.path.join(BASE_DIR, "public_key.pem")

WATERMARK_TEXT = "ASys Encryption"


def load_private_key():
    with open(PRIV_KEY_PATH, "rb") as f:
        data = f.read()

    return serialization.load_pem_private_key(
        data,
        password=None
    )


def load_public_key():
    with open(PUB_KEY_PATH, "rb") as f:
        data = f.read()

    return serialization.load_pem_public_key(data)


def sign_message(msg):
    private_key = load_private_key()

    return private_key.sign(
        msg.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256()
    )


def verify_signature(msg, sig_bytes):
    if sig_bytes is None:
        return False

    try:
        public_key = load_public_key()

        public_key.verify(
            sig_bytes,
            msg.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        return True

    except Exception:
        return False


def bytes_to_bits(data):
    return "".join(f"{byte:08b}" for byte in data)


def bits_to_bytes(bits):
    if not bits:
        return b""

    usable_length = len(bits) // 8 * 8
    bits = bits[:usable_length]

    return bytes(
        int(bits[i:i + 8], 2)
        for i in range(0, len(bits), 8)
    )


def read_cv(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

    elif img.shape[2] == 4:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGRA2BGR
        )

    return img


def decode_lsb(img, bits_len):
    if img is None:
        return None

    flat = img.reshape(-1, img.shape[2])

    if bits_len <= 0:
        return None

    if bits_len % 8 != 0:
        return None

    required_pixels = 32 + bits_len

    if required_pixels > len(flat):
        return None

    bits = "".join(
        str(int(flat[i, 0]) & 1)
        for i in range(32, 32 + bits_len)
    )

    return bits_to_bytes(bits)


def decode_dct(img, bits_len, Q=8.0, bits_per_block=4):
    if img is None:
        return None

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    blocks_y = height // 8
    blocks_x = width // 8

    available_bits = blocks_y * blocks_x * bits_per_block

    if available_bits < bits_len:
        return None

    positions = [
        (4, 4),
        (3, 3),
        (2, 2),
        (5, 5)
    ]

    bits = []
    bit_index = 0

    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            if bit_index >= bits_len:
                break

            y0 = block_y * 8
            x0 = block_x * 8

            block = np.float32(
                gray[y0:y0 + 8, x0:x0 + 8]
            )

            dct_block = cv2.dct(block)

            for row, column in positions:
                if bit_index >= bits_len:
                    break

                quantized_value = int(
                    np.round(dct_block[row, column] / Q)
                )

                bits.append(str(quantized_value & 1))
                bit_index += 1

        if bit_index >= bits_len:
            break

    if len(bits) != bits_len:
        return None

    return bits_to_bytes("".join(bits))


def decode_dwt(img, bits_len, Q=8.0):
    if img is None:
        return None

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    LL, (LH, HL, HH) = pywt.dwt2(
        np.float32(gray),
        "haar"
    )

    flat = LH.flatten()

    if flat.size < bits_len:
        return None

    bits = []

    for i in range(bits_len):
        quantized_value = int(
            np.round(flat[i] / Q)
        )

        bits.append(str(quantized_value & 1))

    return bits_to_bytes("".join(bits))


def save_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(content))


def save_decoded_result(path, decoded_bytes):
    if decoded_bytes is None:
        save_text(path, "Decode failed")
        return

    hexadecimal_result = binascii.hexlify(
        decoded_bytes
    ).decode()

    save_text(path, hexadecimal_result)


def process_blur_folder(folder_name, sig_bits_len):
    print("====================================")
    print(f"Processing blurred folder: {folder_name}")

    folder_path = os.path.join(
        BLUR_DIR,
        folder_name
    )

    decoded_output = os.path.join(
        DECODED_BLUR_BASE_DIR,
        folder_name
    )

    comparison_output = os.path.join(
        COMPARISON_BLUR_BASE_DIR,
        folder_name
    )

    os.makedirs(decoded_output, exist_ok=True)
    os.makedirs(comparison_output, exist_ok=True)

    lsb_valid = False
    dct_valid = False
    dwt_valid = False

    lsb_path = os.path.join(
        folder_path,
        "LSB_blur.png"
    )

    if os.path.exists(lsb_path):
        img = read_cv(lsb_path)

        decoded_bytes = decode_lsb(
            img,
            sig_bits_len
        )

        save_decoded_result(
            os.path.join(
                decoded_output,
                "LSB_blur_decoded.txt"
            ),
            decoded_bytes
        )

        lsb_valid = verify_signature(
            WATERMARK_TEXT,
            decoded_bytes
        )

        print("  LSB valid after blur:", lsb_valid)

    else:
        print("  LSB file not found:", lsb_path)

    dct_path = os.path.join(
        folder_path,
        "DCT_blur.png"
    )

    if os.path.exists(dct_path):
        img = read_cv(dct_path)

        decoded_bytes = decode_dct(
            img,
            sig_bits_len
        )

        save_decoded_result(
            os.path.join(
                decoded_output,
                "DCT_blur_decoded.txt"
            ),
            decoded_bytes
        )

        dct_valid = verify_signature(
            WATERMARK_TEXT,
            decoded_bytes
        )

        print("  DCT valid after blur:", dct_valid)

    else:
        print("  DCT file not found:", dct_path)

    dwt_path = os.path.join(
        folder_path,
        "DWT_blur.png"
    )

    if os.path.exists(dwt_path):
        img = read_cv(dwt_path)

        decoded_bytes = decode_dwt(
            img,
            sig_bits_len
        )

        save_decoded_result(
            os.path.join(
                decoded_output,
                "DWT_blur_decoded.txt"
            ),
            decoded_bytes
        )

        dwt_valid = verify_signature(
            WATERMARK_TEXT,
            decoded_bytes
        )

        print("  DWT valid after blur:", dwt_valid)

    else:
        print("  DWT file not found:", dwt_path)

    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(
        "Blur_Comparison"
    )

    headers = [
        "Folder",
        "LSB valid",
        "DCT valid",
        "DWT valid"
    ]

    for column, header in enumerate(headers):
        worksheet.write(
            0,
            column,
            header
        )

    worksheet.write(1, 0, folder_name)
    worksheet.write(1, 1, str(lsb_valid))
    worksheet.write(1, 2, str(dct_valid))
    worksheet.write(1, 3, str(dwt_valid))

    excel_path = os.path.join(
        comparison_output,
        "comparison_blur.xls"
    )

    workbook.save(excel_path)

    return lsb_valid, dct_valid, dwt_valid


def main():
    try:
        private_key = load_private_key()
        public_key = load_public_key()

    except FileNotFoundError as error:
        print("RSA key file not found:")
        print(error)
        return

    except Exception as error:
        print("Failed to load RSA keys:")
        print(error)
        return

    private_key_size = private_key.key_size
    public_key_size = public_key.key_size

    print("Private key size:", private_key_size)
    print("Public key size:", public_key_size)

    if private_key_size != 2048:
        print("Error: private_key.pem is not RSA-2048.")
        return

    if public_key_size != 2048:
        print("Error: public_key.pem is not RSA-2048.")
        return

    signature_bytes = sign_message(
        WATERMARK_TEXT
    )

    signature_bits_length = len(
        bytes_to_bits(signature_bytes)
    )

    print(
        "Signature bytes:",
        len(signature_bytes)
    )

    print(
        "Signature bits:",
        signature_bits_length
    )

    os.makedirs(
        DECODED_BLUR_BASE_DIR,
        exist_ok=True
    )

    os.makedirs(
        COMPARISON_BLUR_BASE_DIR,
        exist_ok=True
    )

    if not os.path.isdir(BLUR_DIR):
        print("Blur folder not found:", BLUR_DIR)
        return

    folders = sorted(
        folder_name
        for folder_name in os.listdir(BLUR_DIR)
        if os.path.isdir(
            os.path.join(
                BLUR_DIR,
                folder_name
            )
        )
    )

    if not folders:
        print("No blurred folders found.")
        return

    lsb_false = []
    dct_false = []
    dwt_false = []

    for folder_name in folders:
        lsb_ok, dct_ok, dwt_ok = process_blur_folder(
            folder_name,
            signature_bits_length
        )

        if not lsb_ok:
            lsb_false.append(folder_name)

        if not dct_ok:
            dct_false.append(folder_name)

        if not dwt_ok:
            dwt_false.append(folder_name)

    print("====================================")
    print("After blur decode false count:")

    print(
        "LSB false:",
        len(lsb_false),
        lsb_false
    )

    print(
        "DCT false:",
        len(dct_false),
        dct_false
    )

    print(
        "DWT false:",
        len(dwt_false),
        dwt_false
    )

    print("Done.")


if __name__ == "__main__":
    main()