"""
Module F — Step 1: OCR Cache Generation
========================================
Iterates over the 3,687 image-backed malicious samples, runs EasyOCR
on each image, and caches results to EXPERIMENT_CACHE/ocr_cache.csv.

This is a training-time engineering optimization. At inference time,
OCR always runs live. The cache avoids re-running OCR every epoch.

IMMUTABILITY: This script NEVER writes to FINAL_BENCHMARK_DATASET/.
All outputs go to EXPERIMENT_CACHE/.

Usage (Colab — recommended, uses T4 GPU):
    !pip install easyocr
    !python module_f_step1_ocr_cache.py

Usage (Local — CPU, slower):
    uv run --with pandas --with pyarrow --with easyocr module_f_step1_ocr_cache.py
"""

import os
import sys
import warnings

# Suppress PyTorch/dataloader pin_memory and other user warnings to keep terminal output clean
warnings.filterwarnings("ignore", category=UserWarning)

# Force UTF-8 encoding and unbuffered/line-buffered output for standard output/error on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
import time
import pandas as pd
import numpy as np

# Add AGENTS/ to path for ocr_adapter import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "AGENTS"))

from ocr_adapter import OCREngineFactory

# ── Path Configuration ──────────────────────────────────────────────
if os.path.exists("/content/drive/MyDrive/PAS/FINAL_BENCHMARK_DATASET"):
    DATASET_DIR = "/content/drive/MyDrive/PAS/FINAL_BENCHMARK_DATASET"
    CACHE_DIR = "/content/drive/MyDrive/PAS/EXPERIMENT_CACHE"
    USE_GPU = True
else:
    DATASET_DIR = os.path.join(SCRIPT_DIR, "DATASETS", "FINAL_BENCHMARK_DATASET")
    CACHE_DIR = os.path.join(SCRIPT_DIR, "DATASETS", "EXPERIMENT_CACHE")
    USE_GPU = False  # Typically no CUDA on local dev machines

IMAGES_DIR = os.path.join(DATASET_DIR, "IMAGES")
os.makedirs(CACHE_DIR, exist_ok=True)

# OCR Configuration
OCR_ENGINE = "easyocr"
OCR_LANGUAGES = ["en"]  # Add "hi" for Hindi if needed

# ── Step 1: Discover Image-Backed Rows ──────────────────────────────
print("=" * 70)
print("MODULE F — STEP 1: OCR CACHE GENERATION")
print("=" * 70)

print(f"\n[1/5] Loading parquet files to find image-backed rows...")
print(f"      Dataset: {DATASET_DIR}")

train_df = pd.read_parquet(os.path.join(DATASET_DIR, "train.parquet"))
val_df = pd.read_parquet(os.path.join(DATASET_DIR, "validation.parquet"))
test_df = pd.read_parquet(os.path.join(DATASET_DIR, "test.parquet"))

combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Filter to rows with non-null image_path (exactly 3,687 expected)
image_rows = combined[combined["image_path"].notna()].copy()
print(f"      Total rows: {len(combined):,}")
print(f"      Image-backed rows: {len(image_rows):,}")

assert len(image_rows) == 3687, (
    f"Expected exactly 3,687 image-backed rows, got {len(image_rows):,}"
)

# ── Step 2: Resolve Image Paths ─────────────────────────────────────
print(f"\n[2/5] Resolving image paths to local files...")


def resolve_image_path(image_path: str) -> str:
    """
    Resolve an image_path (possibly a Colab Drive absolute path)
    to a local file path.
    """
    # If the path exists as-is (Colab with mounted Drive), use it
    if os.path.exists(image_path):
        return image_path

    # Extract the filename and look in the local IMAGES/ directory
    filename = os.path.basename(image_path)
    local_path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(local_path):
        return local_path

    return None  # Image not found


image_rows["resolved_path"] = image_rows["image_path"].apply(resolve_image_path)

found = image_rows["resolved_path"].notna().sum()
missing = image_rows["resolved_path"].isna().sum()
print(f"      Resolved: {found:,} | Missing: {missing:,}")

if missing > 0:
    missing_files = image_rows[image_rows["resolved_path"].isna()]["image_path"].head(5).tolist()
    print(f"      ⚠️ Sample missing paths: {missing_files}")
    print(f"      Proceeding with {found:,} resolvable images...")

# Filter to resolvable images only
image_rows = image_rows[image_rows["resolved_path"].notna()].copy()

# ── Step 3: Initialize OCR Engine ───────────────────────────────────
print(f"\n[3/5] Initializing {OCR_ENGINE} engine...")
print(f"      Languages: {OCR_LANGUAGES}")
print(f"      GPU: {USE_GPU}")

engine = OCREngineFactory.create(OCR_ENGINE, languages=OCR_LANGUAGES, gpu=USE_GPU)
print(f"      ✅ {engine.name()} initialized")

# ── Step 4: Run OCR on All Images ───────────────────────────────────
print(f"\n[4/5] Running OCR on {len(image_rows):,} images...")

results = []
errors = []
start_time = time.time()

for i, (idx, row) in enumerate(image_rows.iterrows()):
    img_path = row["resolved_path"]
    sample_id = row["sample_id"]
    original_path = row["image_path"]

    img_start = time.time()

    try:
        ocr_text, ocr_confidence = engine.run(img_path)
        img_elapsed = time.time() - img_start

        results.append({
            "image_path": original_path,
            "sample_id": sample_id,
            "ocr_text_cached": ocr_text,
            "ocr_confidence": ocr_confidence,
            "char_count": len(ocr_text),
            "processing_time_sec": round(img_elapsed, 3),
            "error": None,
        })

    except Exception as e:
        img_elapsed = time.time() - img_start
        error_msg = f"{type(e).__name__}: {str(e)}"
        errors.append({"image_path": original_path, "sample_id": sample_id, "error": error_msg})

        results.append({
            "image_path": original_path,
            "sample_id": sample_id,
            "ocr_text_cached": "",
            "ocr_confidence": 0.0,
            "char_count": 0,
            "processing_time_sec": round(img_elapsed, 3),
            "error": error_msg,
        })

    # Progress logging (real-time carriage return updates, with new lines for history every 100 images)
    elapsed = time.time() - start_time
    rate = (i + 1) / elapsed if elapsed > 0 else 0
    eta = (len(image_rows) - (i + 1)) / rate if rate > 0 else 0

    sys.stdout.write(
        f"\r      [{i + 1:>5,} / {len(image_rows):,}] "
        f"{rate:.2f} img/sec | "
        f"elapsed={elapsed:.1f}s | "
        f"ETA={eta:.0f}s | "
        f"errors={len(errors)}"
    )
    sys.stdout.flush()

    if (i + 1) % 100 == 0 or (i + 1) == len(image_rows):
        print()

total_time = time.time() - start_time
print(f"      ✅ OCR complete in {total_time:.1f}s")

# ── Step 5: Save Outputs ───────────────────────────────────────────
print(f"\n[5/5] Saving outputs to:\n      {CACHE_DIR}")

results_df = pd.DataFrame(results)

# Save ocr_cache.csv (the core output: image_path → ocr_text_cached + confidence)
cache_df = results_df[["image_path", "ocr_text_cached", "ocr_confidence"]].copy()
cache_path = os.path.join(CACHE_DIR, "ocr_cache.csv")
cache_df.to_csv(cache_path, index=False)
print(f"      ✅ ocr_cache.csv ({len(cache_df):,} rows)")

# Save full generation log (includes timing, char counts, errors)
log_path = os.path.join(CACHE_DIR, "ocr_generation_log.csv")
results_df.to_csv(log_path, index=False)
print(f"      ✅ ocr_generation_log.csv ({len(results_df):,} rows)")

# ── Summary Statistics ──────────────────────────────────────────────
non_empty = (cache_df["ocr_text_cached"] != "").sum()
mean_len = results_df[results_df["ocr_text_cached"] != ""]["char_count"].mean()
mean_conf = results_df[results_df["ocr_confidence"] > 0]["ocr_confidence"].mean()

print(f"\n{'=' * 70}")
print(f"  SUMMARY")
print(f"  Total images processed: {len(results_df):,}")
print(f"  Non-empty OCR outputs:  {non_empty:,} / {len(results_df):,} "
      f"({non_empty / len(results_df) * 100:.1f}%)")
print(f"  Errors:                 {len(errors)}")
print(f"  Mean text length:       {mean_len:.0f} chars" if not pd.isna(mean_len) else "  Mean text length: N/A")
print(f"  Mean confidence:        {mean_conf:.4f}" if not pd.isna(mean_conf) else "  Mean confidence: N/A")
print(f"  Total processing time:  {total_time:.1f}s "
      f"({len(results_df) / total_time:.1f} img/sec)")
print(f"{'=' * 70}")

if len(errors) > 0:
    print(f"\n⚠️ {len(errors)} errors encountered. Check ocr_generation_log.csv for details.")

all_passed = non_empty >= int(len(results_df) * 0.99)  # ≥99% non-empty
if all_passed:
    print("\n✅ MODULE F STEP 1 COMPLETE — OCR cache generated successfully.")
else:
    print(f"\n⚠️ MODULE F STEP 1 — Only {non_empty / len(results_df) * 100:.1f}% "
          f"non-empty (expected ≥99%). Review error log.")

sys.exit(0 if all_passed else 1)
