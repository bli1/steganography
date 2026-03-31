import os
import cv2

def pick_base_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if os.path.isdir(os.path.join(here, "Encoded_image")):
        return here
    if os.path.isdir(os.path.join(parent, "Encoded_image")):
        return parent
    return parent

BASE_DIR = pick_base_dir()
ENCODED_DIR = os.path.join(BASE_DIR, "Encoded_image")
JPEG_BASE_DIR = os.path.join(BASE_DIR, "Encoded_jpeg")

JPEG_QUALITIES = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]

def main():
    if not os.path.isdir(ENCODED_DIR):
        print("Encoded_image folder not found:", ENCODED_DIR)
        return

    folders = sorted([
        f for f in os.listdir(ENCODED_DIR)
        if os.path.isdir(os.path.join(ENCODED_DIR, f))
    ])

    if not folders:
        print("No encoded folders found.")
        return

    os.makedirs(JPEG_BASE_DIR, exist_ok=True)

    for q in JPEG_QUALITIES:
        q_dir = os.path.join(JPEG_BASE_DIR, f"Q{q}")
        os.makedirs(q_dir, exist_ok=True)

        for folder in folders:
            src_dir = os.path.join(ENCODED_DIR, folder)
            dst_dir = os.path.join(q_dir, folder)
            os.makedirs(dst_dir, exist_ok=True)

            for name in ["LSB.png", "DCT.png", "DWT.png"]:
                src = os.path.join(src_dir, name)
                if not os.path.exists(src):
                    continue

                img = cv2.imread(src, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                out_name = name.replace(".png", ".jpg")
                out_path = os.path.join(dst_dir, out_name)
                cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])

    print("All JPEG-compressed images generated.")

if __name__ == "__main__":
    main()
