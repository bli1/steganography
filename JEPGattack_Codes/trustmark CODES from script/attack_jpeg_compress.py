from pathlib import Path
from PIL import Image

ROOT = Path("..")
INPUT_DIR = ROOT / "TrustMark_encoded"
BASE_OUT_DIR = ROOT / "TrustMark_jpeg"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
JPEG_QUALITIES = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]

def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def main():
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Input folder not found: {INPUT_DIR.resolve()}")
    images = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not images:
        print("No images found in:", INPUT_DIR.resolve())
        return
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for q in JPEG_QUALITIES:
        out_dir = BASE_OUT_DIR / f"Q{q}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in images:
            img = load_rgb(p)
            out_path = out_dir / (p.stem + ".jpg")
            img.save(out_path, format="JPEG", quality=int(q), optimize=True)
    print("All TrustMark JPEG-compressed images generated.")

if __name__ == "__main__":
    main()
