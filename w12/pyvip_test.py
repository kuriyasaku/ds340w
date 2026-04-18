import os
import time
import pandas as pd
import pyvips


MODE = "benign"
CSV_PATH = "benign_top5_all.csv"
TIF_DIR = "."
OUTPUT_ROOT = "output"

PATCH_SIZE = 256

# Only test this one slide first
TARGET_SLIDE_ID = "normal_002"

ENABLE_RANGE_FILTER = False
START_SLIDE_NUM = 4
END_SLIDE_NUM = 60

# Run fewer patches first to make sure pyvips, coordinates,
# and output files are all working correctly
LIMIT_PATCHES_PER_SLIDE = None
ATTENTION_DECIMALS = 6


# =========================
# Utility functions
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
# Main logic
# =========================

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    if not os.path.exists(TIF_DIR):
        raise FileNotFoundError(f"TIFF directory not found: {TIF_DIR}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_cols = ["slide_id", "x", "y", "attention_raw", "rank"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["slide_id"] = df["slide_id"].astype(str)
    df["rank"] = df["rank"].apply(safe_int)

    # Keep only the target slide
    df = df[df["slide_id"] == TARGET_SLIDE_ID].copy()

    if ENABLE_RANGE_FILTER:
        df = df[df["slide_id"].apply(lambda s: slide_in_range(s, START_SLIDE_NUM, END_SLIDE_NUM))].copy()

    if df.empty:
        raise ValueError(f"No data found after filtering. Please check whether {TARGET_SLIDE_ID} exists in the CSV.")

    class_output_dir = os.path.join(OUTPUT_ROOT, MODE)
    os.makedirs(class_output_dir, exist_ok=True)

    slide_ids = sorted(df["slide_id"].unique())

    print("=" * 80, flush=True)
    print(f"Mode: {MODE}", flush=True)
    print(f"CSV: {CSV_PATH}", flush=True)
    print(f"TIFF directory: {TIF_DIR}", flush=True)
    print(f"Output root directory: {OUTPUT_ROOT}", flush=True)
    print(f"Class output directory: {class_output_dir}", flush=True)
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}", flush=True)
    print(f"Target slide: {TARGET_SLIDE_ID}", flush=True)
    print(f"Number of slides to process: {len(slide_ids)}", flush=True)
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
        print(f"[Slide {slide_index}/{len(slide_ids)}] Start processing {slide_id}", flush=True)
        print(f"Matching TIFF: {tif_path}", flush=True)
        print(f"Number of patches in this slide: {len(slide_df)}", flush=True)

        if not os.path.exists(tif_path):
            print(f"[Warning] TIFF not found, skipping: {tif_path}", flush=True)
            total_missing_tif += 1
            continue

        slide_output_dir = os.path.join(class_output_dir, slide_id)
        os.makedirs(slide_output_dir, exist_ok=True)
        print(f"Output folder: {slide_output_dir}", flush=True)

        slide_success = 0
        slide_skipped = 0

        img = pyvips.Image.new_from_file(tif_path, access="random")

        width = img.width
        height = img.height
        print(f"TIFF size: width={width}, height={height}", flush=True)

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
                f"Processing: {filename} | box=({left}, {top}, {left + patch_w}, {top + patch_h})",
                flush=True
            )

            if left < 0 or top < 0 or left + patch_w > width or top + patch_h > height:
                print(
                    f"[{slide_id}] [{idx + 1}/{len(slide_df)}] "
                    f"Skipped, out of bounds",
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
                    f"Done: {filename} | Time used: {dt:.2f}s",
                    flush=True
                )
            except Exception as e:
                print(
                    f"[{slide_id}] [{idx + 1}/{len(slide_df)}] "
                    f"Failed: {filename} | Error: {e}",
                    flush=True
                )
                slide_skipped += 1
                total_skipped += 1

        print(f"[Slide done] {slide_id}", flush=True)
        print(f"Successfully exported: {slide_success}", flush=True)
        print(f"Skipped: {slide_skipped}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("All processing finished", flush=True)
    print(f"Total successfully exported: {total_success}", flush=True)
    print(f"Total skipped: {total_skipped}", flush=True)
    print(f"Missing TIFF count: {total_missing_tif}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
