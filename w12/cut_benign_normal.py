import os
import math
import pandas as pd
from PIL import Image, ImageFile

# 允许处理超大图
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 可改参数
# =========================
CSV_PATH = "benign_top5_all.csv"
TIF_PATH = "normal_002.tif"
OUTPUT_DIR = "benign_cut_output"

TARGET_SLIDE_ID = "normal_002"
PATCH_SIZE = 256

# True = 只切 TARGET_SLIDE_ID
# False = csv 里有什么 slide_id 就切什么
FILTER_BY_SLIDE = True


def safe_int(v):
    return int(round(float(v)))


def format_attention(v):
    return f"{float(v):.6f}"


def build_filename(slide_id, rank, x, y, attention_raw):
    return (
        f"{slide_id}"
        f"_top{int(rank):05d}"
        f"_x_{int(x)}"
        f"_y_{int(y)}"
        f"_a_{format_attention(attention_raw)}.png"
    )


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV 文件: {CSV_PATH}")

    if not os.path.exists(TIF_PATH):
        raise FileNotFoundError(f"找不到 TIFF 文件: {TIF_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "slide_id", "x", "y", "attention_raw", "rank"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    if FILTER_BY_SLIDE:
        df = df[df["slide_id"].astype(str) == TARGET_SLIDE_ID].copy()

    if df.empty:
        raise ValueError("筛选后没有任何行，请检查 slide_id 或 CSV 内容。")

    # 排序，保证 top00001, top00002...
    df["rank"] = df["rank"].apply(safe_int)
    df = df.sort_values(by="rank").reset_index(drop=True)

    print(f"准备打开 TIFF: {TIF_PATH}")
    img = Image.open(TIF_PATH)

    width, height = img.size
    print(f"TIFF 尺寸: width={width}, height={height}")
    print(f"准备切图数量: {len(df)}")
    print(f"patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"输出目录: {OUTPUT_DIR}")

    success = 0
    skipped = 0

    for idx, row in df.iterrows():
        slide_id = str(row["slide_id"])
        x = safe_int(row["x"])
        y = safe_int(row["y"])
        rank = safe_int(row["rank"])
        attention_raw = float(row["attention_raw"])

        # 左上角 + 256
        left = x
        upper = y
        right = x + PATCH_SIZE
        lower = y + PATCH_SIZE

        filename = build_filename(slide_id, rank, x, y, attention_raw)
        save_path = os.path.join(OUTPUT_DIR, filename)

        print(
            f"[{idx + 1}/{len(df)}] 正在处理: {filename} | "
            f"box=({left}, {upper}, {right}, {lower})",
            flush=True
        )

        # 越界检查
        if left < 0 or upper < 0 or right > width or lower > height:
            print(
                f"[跳过] rank={rank}, x={x}, y={y} 超出边界 "
                f"(box=({left}, {upper}, {right}, {lower}), image=({width}, {height}))",
                flush=True
            )
            skipped += 1
            continue

        patch = img.crop((left, upper, right, lower))

        if patch.mode not in ("RGB", "L"):
            patch = patch.convert("RGB")

        patch.save(save_path, format="PNG")

        success += 1

        print(f"[完成] 已保存到: {save_path}", flush=True)

    print("\n处理完成")
    print(f"成功导出: {success}")
    print(f"跳过数量: {skipped}")


if __name__ == "__main__":
    main()