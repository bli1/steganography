import os
import cv2
import numpy as np
from PIL import Image, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 第一个方法的输出
ENCODED_DIR = os.path.join(BASE_DIR, "Encoded_image")

# blur 后输出（给 blur-decode 用）
BLUR_DIR = os.path.join(BASE_DIR, "Encoded_blur")
os.makedirs(BLUR_DIR, exist_ok=True)

BLUR_RADIUS = 0.5   # 和你后面 decode 保持一致


def blur_cv2_bgr_with_pillow(img, r):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil2 = pil.filter(ImageFilter.GaussianBlur(radius=r))
    return cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)


def main():
    if not os.path.isdir(ENCODED_DIR):
        print("Encoded_image folder not found:", ENCODED_DIR)
        return

    folders = [
        f for f in os.listdir(ENCODED_DIR)
        if os.path.isdir(os.path.join(ENCODED_DIR, f))
    ]
    folders.sort()

    if not folders:
        print("No encoded folders found.")
        return

    for folder in folders:
        print("Blurring:", folder)
        src_dir = os.path.join(ENCODED_DIR, folder)
        dst_dir = os.path.join(BLUR_DIR, folder)
        os.makedirs(dst_dir, exist_ok=True)

        for name in ["LSB.png", "DCT.png", "DWT.png"]:
            src = os.path.join(src_dir, name)
            if not os.path.exists(src):
                continue

            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                continue

            blur_img = blur_cv2_bgr_with_pillow(img, BLUR_RADIUS)
            out_name = name.replace(".png", "_blur.png")
            cv2.imwrite(os.path.join(dst_dir, out_name), blur_img)

    print("All blur images generated.")


if __name__ == "__main__":
    main()
