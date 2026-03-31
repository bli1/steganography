import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始 encode 后文件夹
ENCODED_DIR = os.path.join(BASE_DIR, "Encoded_image")

# 缩放攻击后的输出（给第 4 个脚本读）
GEO_DIR = os.path.join(BASE_DIR, "Encoded_geo")
os.makedirs(GEO_DIR, exist_ok=True)

# ========= 只做缩放攻击参数 =========
SCALE = 0.8                 # 缩放比例（<1 缩小，>1 放大）
INTERP = cv2.INTER_LINEAR   # 插值方式
BORDER = cv2.BORDER_REFLECT_101


def scale_attack(img_bgr: np.ndarray) -> np.ndarray:

    h, w = img_bgr.shape[:2]

    # 先缩放
    new_w = max(1, int(w * SCALE))
    new_h = max(1, int(h * SCALE))
    scaled = cv2.resize(img_bgr, (new_w, new_h), interpolation=INTERP)

    # 再拉回原尺寸
    attacked = cv2.resize(scaled, (w, h), interpolation=INTERP)

    return attacked


def main():
    if not os.path.isdir(ENCODED_DIR):
        print("Encoded_image folder not found:", ENCODED_DIR)
        return

    folders = sorted(
        f for f in os.listdir(ENCODED_DIR)
        if os.path.isdir(os.path.join(ENCODED_DIR, f))
    )
    if not folders:
        print("No encoded folders found.")
        return

    for folder in folders:
        print("Scaling attack:", folder)
        src_dir = os.path.join(ENCODED_DIR, folder)
        dst_dir = os.path.join(GEO_DIR, folder)
        os.makedirs(dst_dir, exist_ok=True)


        for name in ["LSB.png", "DCT.png", "DWT.png"]:
            src = os.path.join(src_dir, name)
            if not os.path.exists(src):
                continue

            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                continue

            attacked_img = scale_attack(img)

            # 输出命名：LSB_geo.png / DCT_geo.png / DWT_geo.png
            out_name = name.replace(".png", "_geo.png")
            cv2.imwrite(os.path.join(dst_dir, out_name), attacked_img)

    print("All scaling-attack images generated.")
    print("Output folder:", GEO_DIR)


if __name__ == "__main__":
    main()