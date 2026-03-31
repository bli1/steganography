import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "Spread_encoded")
DST_BASE_DIR = os.path.join(BASE_DIR, "Spread_jpeg")
JPEG_QUALITIES = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]

def main():
    if not os.path.isdir(SRC_DIR):
        print("Spread_encoded folder not found:", SRC_DIR)
        return
    folders = sorted([f for f in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, f))])
    if not folders:
        print("No spread encoded folders found.")
        return
    os.makedirs(DST_BASE_DIR, exist_ok=True)
    for q in JPEG_QUALITIES:
        q_dir = os.path.join(DST_BASE_DIR, f"Q{q}")
        os.makedirs(q_dir, exist_ok=True)
        for folder in folders:
            src_path = os.path.join(SRC_DIR, folder, "spread.png")
            if not os.path.exists(src_path):
                continue
            img = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            dst_dir = os.path.join(q_dir, folder)
            os.makedirs(dst_dir, exist_ok=True)
            out_path = os.path.join(dst_dir, "spread.jpg")
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    print("All DWT-SS JPEG-compressed images generated.")

if __name__ == "__main__":
    main()
