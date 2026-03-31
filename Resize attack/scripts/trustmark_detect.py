   # trustmark_detect.py
# =========================
# TrustMark Detection (Decode) - Full Script
# Usage:
#   1) Put this file under: 科研2/scripts/
#   2) Set DETECT_DIR_NAME to "TrustMark_encoded" or "TrustMark_blur"
#   3) python trustmark_detect.py
# =========================

from pathlib import Path
from PIL import Image
from trustmark import TrustMark


# ====== 路径设置（scripts 的上一级是科研2） ======
ROOT = Path("..")


DETECT_DIR_NAME = "TrustMark_geo"
DETECT_DIR = ROOT / DETECT_DIR_NAME

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


EXPECTED_COMMIT = "ZER7C7BQ"   # 不想校验就改成 None


def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def detect_one(tm: TrustMark, img_path: Path):
    """返回 (secret, present, schema, error_msg)"""
    try:
        img = load_rgb(img_path)
        secret, present, schema = tm.decode(img)
        return secret, present, schema, None
    except Exception as e:
        return None, False, None, str(e)


def main():
    if not DETECT_DIR.exists():
        raise RuntimeError(f"找不到检测目录: {DETECT_DIR.resolve()}")

    images = list_images(DETECT_DIR)
    print(f"[OK] Detection folder: {DETECT_DIR.resolve()}")
    print(f"[OK] Images: {len(images)}")
    if EXPECTED_COMMIT is not None:
        print(f"[OK] EXPECTED_COMMIT: {EXPECTED_COMMIT}")

    tm = TrustMark()

    present_cnt = 0
    match_cnt = 0
    fail_cnt = 0

    for p in images:
        secret, present, schema, err = detect_one(tm, p)

        if err is not None:
            fail_cnt += 1
            print(f"[ERR] {p.name:25s} | decode failed: {err}")
            continue

        if present:
            present_cnt += 1

        match = None
        if EXPECTED_COMMIT is not None:
            match = bool(present and secret == EXPECTED_COMMIT)
            if match:
                match_cnt += 1

        # 输出一行结果
        line = f"{p.name:25s} | present={present} | secret={secret} | schema={schema}"
        if match is not None:
            line += f" | match={match}"
        #print(line)

    # Summary
    print("\n===== Summary =====")
    print(f"Folder            : {DETECT_DIR_NAME}")
    print(f"Total images      : {len(images)}")
    print(f"Decode failed     : {fail_cnt}")
    print(f"Watermark present : {present_cnt} ({present_cnt}/{len(images)})")
    if EXPECTED_COMMIT is not None:
        print(f"Payload matched   : {match_cnt} ({match_cnt}/{len(images)})")

    # 可选：保存 CSV（方便你后面做 benchmark 表）
    out_csv = ROOT / "TrustMark_detect_result.csv"
    try:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["folder", "image", "present", "secret", "schema", "match", "error"])
            # 再跑一遍，把每张图的细节写进 CSV（为了结构清晰）
            # 如果你想效率更高，也可以把循环里结果缓存起来再写
            for p in images:
                secret, present, schema, err = detect_one(tm, p)
                if err is not None:
                    writer.writerow([DETECT_DIR_NAME, p.name, False, "", "", "", err])
                else:
                    match = ""
                    if EXPECTED_COMMIT is not None:
                        match = (present and secret == EXPECTED_COMMIT)
                    writer.writerow([DETECT_DIR_NAME, p.name, present, secret, schema, match, ""])
        print(f"[OK] CSV saved: {out_csv.resolve()}")
    except Exception as e:
        print(f"[WARN] CSV 保存失败: {e}")


if __name__ == "__main__":
    main()
