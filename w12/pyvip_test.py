import os
import time
import pandas as pd
import pyvips


MODE = "benign"
CSV_PATH = "benign_top5_all.csv"
TIF_DIR = "."
OUTPUT_ROOT = "output"

PATCH_SIZE = 256

# 只测试这一张
TARGET_SLIDE_ID = "normal_003"

ENABLE_RANGE_FILTER = False
START_SLIDE_NUM = 4
END_SLIDE_NUM = 60

# 先少跑一点，确认 pyvips、坐标、输出都正常
LIMIT_PATCHES_PER_SLIDE = None
ATTENTION_DECIMALS = 6


# =========================
# 工具函数
# =========================

def safe_int(v):
    return int(round(float(v)))


def format_attention(v):
    return f"{float(v):.{ATTENTION_DECIMALS}f}"


def build_filename(slide_id, rank, x, y, attention_raw):
    return (
        f"{slide_id}"
        f"_top{int(rank):05d}"
        f"_x_{int(x)}"
        f"_y_{int(y)}"
        f"_a_{format_attention(attention_raw)}.png"
    )


def extract_slide_num(slide_id: str):
    try:
        return int(slide_id.split("_")[-1])
    except Exception:
        return None


def slide_in_range(slide_id: str, start_num: int, end_num: int):
    n = extract_slide_num(slide_id)
    if n is None:
        return False
    return start_num <= n <= end_num


def get_tif_path(slide_id: str):
    return os.path.join(TIF_DIR, f"{slide_id}.tif")


def ensure_uint8_rgb(img: pyvips.Image) -> pyvips.Image:
    if img.format != "uchar":
        img = img.cast("uchar")

    if img.bands == 1:
        img = img.bandjoin([img, img])
    elif img.bands > 3:
        img = img[:3]

    return img


# =========================
# 主逻辑
# =========================

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV 文件: {CSV_PATH}")

    if not os.path.exists(TIF_DIR):
        raise FileNotFoundError(f"找不到 TIFF 目录: {TIF_DIR}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_cols = ["slide_id", "x", "y", "attention_raw", "rank"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    df["slide_id"] = df["slide_id"].astype(str)
    df["rank"] = df["rank"].apply(safe_int)

    # 只保留 normal_001
    df = df[df["slide_id"] == TARGET_SLIDE_ID].copy()

    if ENABLE_RANGE_FILTER:
        df = df[df["slide_id"].apply(lambda s: slide_in_range(s, START_SLIDE_NUM, END_SLIDE_NUM))].copy()

    if df.empty:
        raise ValueError(f"筛选后没有数据，请检查 CSV 中是否有 {TARGET_SLIDE_ID}")

    class_output_dir = os.path.join(OUTPUT_ROOT, MODE)
    os.makedirs(class_output_dir, exist_ok=True)

    slide_ids = sorted(df["slide_id"].unique())

    print("=" * 80, flush=True)
    print(f"模式: {MODE}", flush=True)
    print(f"CSV: {CSV_PATH}", flush=True)
    print(f"TIF 目录: {TIF_DIR}", flush=True)
    print(f"输出根目录: {OUTPUT_ROOT}", flush=True)
    print(f"分类输出目录: {class_output_dir}", flush=True)
    print(f"patch size: {PATCH_SIZE} x {PATCH_SIZE}", flush=True)
    print(f"目标 slide: {TARGET_SLIDE_ID}", flush=True)
    print(f"待处理 slide 数量: {len(slide_ids)}", flush=True)
    print("=" * 80, flush=True)

    total_success = 0
    total_skipped = 0
    total_missing_tif = 0

    for slide_index, slide_id in enumerate(slide_ids, start=1):
        slide_df = df[df["slide_id"] == slide_id].copy()
        slide_df = slide_df.sort_values(by="rank").reset_index(drop=True)

        if LIMIT_PATCHES_PER_SLIDE is not None:
            slide_df = slide_df.head(LIMIT_PATCHES_PER_SLIDE)

        tif_path = get_tif_path(slide_id)

        print("\n" + "-" * 80, flush=True)
        print(f"[Slide {slide_index}/{len(slide_ids)}] 开始处理 {slide_id}", flush=True)
        print(f"对应 TIFF: {tif_path}", flush=True)
        print(f"该 slide patch 数量: {len(slide_df)}", flush=True)

        if not os.path.exists(tif_path):
            print(f"[警告] 找不到 TIFF，跳过: {tif_path}", flush=True)
            total_missing_tif += 1
            continue

        slide_output_dir = os.path.join(class_output_dir, slide_id)
        os.makedirs(slide_output_dir, exist_ok=True)
        print(f"输出文件夹: {slide_output_dir}", flush=True)

        slide_success = 0
        slide_skipped = 0

        img = pyvips.Image.new_from_file(tif_path, access="random")

        width = img.width
        height = img.height
        print(f"TIFF 尺寸: width={width}, height={height}", flush=True)

        for idx, row in slide_df.iterrows():
            t0 = time.time()

            x = safe_int(row["x"])
            y = safe_int(row["y"])
            rank = safe_int(row["rank"])
            attention_raw = float(row["attention_raw"])

            left = x
            top = y
            patch_w = PATCH_SIZE
            patch_h = PATCH_SIZE

            filename = build_filename(slide_id, rank, x, y, attention_raw)
            save_path = os.path.join(slide_output_dir, filename)

            print(
                f"[{slide_id}] [{idx + 1}/{len(slide_df)}] "
                f"开始处理: {filename} | box=({left}, {top}, {left + patch_w}, {top + patch_h})",
                flush=True
            )

            if left < 0 or top < 0 or left + patch_w > width or top + patch_h > height:
                print(
                    f"[{slide_id}] [{idx + 1}/{len(slide_df)}] "
                    f"跳过，超出边界",
                    flush=True
                )
                slide_skipped += 1
                total_skipped += 1
                continue

            try:
                patch = img.crop(left, top, patch_w, patch_h)
                patch = ensure_uint8_rgb(patch)
                patch.pngsave(save_path)

                dt = time.time() - t0
                slide_success += 1
                total_success += 1

                print(
                    f"[{slide_id}] [{idx + 1}/{len(slide_df)}] "
                    f"完成: {filename} | 用时 {dt:.2f}s",
                    flush=True
                )
            except Exception as e:
                print(
                    f"[{slide_id}] [{idx + 1}/{len(slide_df)}] "
                    f"失败: {filename} | 错误: {e}",
                    flush=True
                )
                slide_skipped += 1
                total_skipped += 1

        print(f"[Slide 完成] {slide_id}", flush=True)
        print(f"成功导出: {slide_success}", flush=True)
        print(f"跳过数量: {slide_skipped}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("全部处理完成", flush=True)
    print(f"总成功导出: {total_success}", flush=True)
    print(f"总跳过数量: {total_skipped}", flush=True)
    print(f"缺失 TIFF 数量: {total_missing_tif}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()