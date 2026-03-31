from pathlib import Path
from PIL import Image, ImageFilter

# ============ 路径 ============
ROOT = Path(__file__).resolve().parent.parent

# 输入：TrustMark embed 后的图
INPUT_DIR = ROOT / "TrustMark_encoded"

# 输出：blur 后的图（自动编号，避免覆盖）
BASE_OUT_DIR = ROOT / "TrustMark_blur"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ============ 攻击参数 ============
SIGMA = 2.0   # 和你之前实验保持一致（例如 1.0 / 2.0）


def make_unique_dir(base_dir: Path) -> Path:
    """避免覆盖：TrustMark_blur, TrustMark_blur_1, ..."""
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    i = 1
    while True:
        cand = base_dir.parent / f"{base_dir.name}_{i}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        i += 1


def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def main():
    if not INPUT_DIR.exists():
        raise RuntimeError(f"找不到输入目录: {INPUT_DIR.resolve()}")

    images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"[OK] Gaussian Blur 攻击，σ={SIGMA}")
    print(f"[OK] 输入图片数: {len(images)}")

    out_dir = make_unique_dir(BASE_OUT_DIR)
    print(f"[OK] 输出目录: {out_dir.resolve()}")

    for p in images:
        img = load_rgb(p)

        # Gaussian Blur（PIL）
        img_blur = img.filter(ImageFilter.GaussianBlur(radius=SIGMA))

        out_path = out_dir / p.name
        img_blur.save(out_path)

        print(f"  {p.name} -> {out_path.name}")

    print("[DONE] Gaussian blur attack finished.")


if __name__ == "__main__":
    main()
