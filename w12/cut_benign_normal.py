import os
import math
import pandas as pd
from PIL import Image, ImageFile

# Allow processing of very large images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# Editable parameters
# =========================
CSV_PATH = "benign_top5_all.csv"
TIF_PATH = "normal_002.tif"
OUTPUT_DIR = "benign_cut_output"

TARGET_SLIDE_ID = "normal_002"
PATCH_SIZE = 256

# True = only cut TARGET_SLIDE_ID
# False = cut whatever slide_id appears in the CSV
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
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    if not os.path.exists(TIF_PATH):
        raise FileNotFoundError(f"TIFF file not found: {TIF_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "slide_id", "x", "y", "attention_raw", "rank"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if FILTER_BY_SLIDE:
        df = df[df["slide_id"].astype(str) == TARGET_SLIDE_ID].copy()

    if df.empty:
        raise ValueError("No rows found after filtering. Please check the slide_id or CSV content.")

    # Sort to make sure filenames follow top00001, top00002, ...
    df["rank"] = df["rank"].apply(safe_int)
    df = df.sort_values(by="rank").reset_index(drop=True)

    print(f"Preparing to open TIFF: {TIF_PATH}")
    img = Image.open(TIF_PATH)

    width, height = img.size
    print(f"TIFF size: width={width}, height={height}")
    print(f"Number of patches to cut: {len(df)}")
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")

    success = 0
    skipped = 0

    for idx, row in df.iterrows():
        slide_id = str(row["slide_id"])
        x = safe_int(row["x"])
        y = safe_int(row["y"])
        rank = safe_int(row["rank"])
        attention_raw = float(row["attention_raw"])

        # Top-left corner + 256
        left = x
        upper = y
        right = x + PATCH_SIZE
        lower = y + PATCH_SIZE

        filename = build_filename(slide_id, rank, x, y, attention_raw)
        save_path = os.path.join(OUTPUT_DIR, filename)

        print(
            f"[{idx + 1}/{len(df)}] Processing: {filename} | "
            f"box=({left}, {upper}, {right}, {lower})",
            flush=True
        )

        # Boundary check
        if left < 0 or upper < 0 or right > width or lower > height:
            print(
                f"[Skipped] rank={rank}, x={x}, y={y} is out of bounds "
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

        print(f"[Done] Saved to: {save_path}", flush=True)

    print("\nProcessing finished")
    print(f"Successfully exported: {success}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
