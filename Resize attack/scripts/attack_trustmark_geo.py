

from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import sys

ROOT = Path("..")
INPUT_DIR = ROOT / "TrustMark_encoded"
OUT_DIR = ROOT / "TrustMark_geo"

SCALE = 0.8
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def scale_attack(img):
    h, w = img.shape[:2]
    new_w = int(w * SCALE)
    new_h = int(h * SCALE)

    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    attacked = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return attacked


def main():
    if not INPUT_DIR.exists():
        print("找不到 TrustMark_encoded")
        return

    OUT_DIR.mkdir(exist_ok=True)

    imgs = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]

    print("Geo attack (scale only)")
    print("scale =", SCALE)

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue

        attacked = scale_attack(img)

        out_name = p.stem + "_geo.png"
        cv2.imwrite(str(OUT_DIR / out_name), attacked)
        print("done:", p.name)


if __name__ == "__main__":
    main()