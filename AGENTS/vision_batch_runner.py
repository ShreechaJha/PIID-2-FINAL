"""
Vision Batch Runner — Colab-Optimized Batch Feature Extraction
===============================================================

Runs the VisionAgent across ALL 289,874 samples in the benchmark dataset:
1. TextAnalyzer on ALL samples (fast, ~2 min)
2. ImageAnalyzer on ~3,687 samples with images (slower, ~30-60 min on T4)
3. Merges results, computes vision_score for every sample
4. Saves checkpoint every N images for Colab session resilience
5. Outputs: DATASETS/EXPERIMENT_CACHE/vision_features.csv

Usage in Colab:
    # Mount drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Add agents to path
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/PIID_PROJECT/AGENTS')

    # Run
    from vision_batch_runner import run_batch_extraction
    run_batch_extraction(
        project_root='/content/drive/MyDrive/PIID_PROJECT',
        gpu=True
    )
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Ensure the AGENTS directory is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from vision_agent import VisionAgent, TextAnalyzer, ImageAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_paths(project_root: str) -> dict:
    """Resolve all required file paths from project root."""
    root = Path(project_root)
    ds = root / "DATASETS" / "FINAL_BENCHMARK_DATASET"
    cache = root / "DATASETS" / "EXPERIMENT_CACHE"
    cache.mkdir(parents=True, exist_ok=True)

    return {
        "benchmark_text": ds / "benchmark_text.csv",
        "metadata": ds / "metadata.csv",
        "labels": ds / "benchmark_labels.csv",
        "images_dir": ds / "IMAGES",
        "corrected_splits": cache / "corrected_splits.csv",
        "output_csv": cache / "vision_features.csv",
        "image_checkpoint": cache / "vision_image_checkpoint.csv",
        "text_checkpoint": cache / "vision_text_features.csv",
        "run_log": cache / "vision_batch_log.json",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Text Feature Extraction (ALL samples)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_features(
    benchmark_text_path: str,
    output_path: str,
    text_col: str = "text",
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Run TextAnalyzer on every sample's text column.

    Args:
        benchmark_text_path: Path to benchmark_text.csv.
        output_path: Path to save text features checkpoint.
        text_col: Column name containing text (default: 'text').
        force_rerun: If True, ignore existing checkpoint.

    Returns:
        DataFrame with sample_id + text features.
    """
    # Check for existing checkpoint
    if os.path.exists(output_path) and not force_rerun:
        logger.info("Loading existing text features from %s", output_path)
        return pd.read_csv(output_path)

    logger.info("Loading benchmark_text.csv from %s ...", benchmark_text_path)
    # Read only needed columns to save memory
    df = pd.read_csv(
        benchmark_text_path,
        usecols=["sample_id", text_col],
        dtype={"sample_id": str},
        low_memory=False,
    )
    logger.info("Loaded %d samples", len(df))

    # Fill missing text
    df[text_col] = df[text_col].fillna("")

    # Initialize analyzer
    analyzer = TextAnalyzer()
    feature_cols = list(TextAnalyzer.default_features().keys())

    logger.info("Extracting text features for %d samples...", len(df))
    t0 = time.time()

    # Vectorized-ish batch processing (apply is faster than iterrows)
    results = df[text_col].apply(analyzer.analyze)

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(results.tolist())

    # Combine with sample_id
    text_features = pd.concat([df[["sample_id"]].reset_index(drop=True),
                               features_df.reset_index(drop=True)], axis=1)

    elapsed = time.time() - t0
    logger.info(
        "Text features extracted in %.1f seconds (%.0f samples/sec)",
        elapsed, len(df) / elapsed,
    )

    # Quick stats
    mal_mask = text_features["keyword_density"] > 0
    logger.info(
        "Samples with keyword_density > 0: %d / %d (%.1f%%)",
        mal_mask.sum(), len(text_features),
        100.0 * mal_mask.sum() / len(text_features),
    )

    # Save checkpoint
    text_features.to_csv(output_path, index=False)
    logger.info("Text features saved to %s", output_path)

    return text_features


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Image Feature Extraction (samples with images only)
# ─────────────────────────────────────────────────────────────────────────────

def extract_image_features(
    metadata_path: str,
    images_dir: str,
    checkpoint_path: str,
    gpu: bool = True,
    checkpoint_every: int = 200,
    force_rerun: bool = False,
    run_watermark: bool = False,
) -> pd.DataFrame:
    """
    Run ImageAnalyzer on samples that have images.

    Includes checkpoint/resume logic for Colab session resilience.

    Args:
        metadata_path: Path to metadata.csv.
        images_dir: Path to IMAGES directory in benchmark.
        checkpoint_path: Path to save/resume image feature checkpoint.
        gpu: Use GPU for EasyOCR.
        checkpoint_every: Save checkpoint every N images.
        force_rerun: If True, ignore existing checkpoint.
        run_watermark: Run watermark detection (slower).

    Returns:
        DataFrame with sample_id + image features.
    """
    # Load metadata to find samples with images
    logger.info("Loading metadata from %s ...", metadata_path)
    meta = pd.read_csv(
        metadata_path,
        usecols=["sample_id", "image_path"],
        dtype={"sample_id": str, "image_path": str},
        low_memory=False,
    )

    # Filter to samples with image_path set
    meta["image_path"] = meta["image_path"].fillna("")
    has_image = meta[meta["image_path"].str.strip().str.len() > 0].copy()
    logger.info("Found %d samples with image_path in metadata", len(has_image))

    if has_image.empty:
        logger.warning("No samples with images found!")
        return pd.DataFrame(columns=["sample_id"])

    # Extract just the filename from the full Drive path
    # metadata stores paths like: /content/drive/MyDrive/PAS/FINAL_.../IMAGES/abc.png
    # We need to resolve to local images_dir
    def resolve_image(row_path):
        if not row_path:
            return None
        basename = os.path.basename(row_path.strip())
        local_path = os.path.join(images_dir, basename)
        if os.path.exists(local_path):
            return local_path
        return None

    has_image["local_image_path"] = has_image["image_path"].apply(resolve_image)
    valid_images = has_image[has_image["local_image_path"].notna()].copy()
    logger.info("Resolved %d images to local paths", len(valid_images))

    if valid_images.empty:
        logger.warning("No images could be resolved to local paths!")
        return pd.DataFrame(columns=["sample_id"])

    # Check for existing checkpoint (resume support)
    processed_ids = set()
    existing_rows = []
    if os.path.exists(checkpoint_path) and not force_rerun:
        existing = pd.read_csv(checkpoint_path)
        processed_ids = set(existing["sample_id"].tolist())
        existing_rows = existing.to_dict("records")
        logger.info(
            "Resuming from checkpoint: %d images already processed",
            len(processed_ids),
        )

    # Filter out already-processed samples
    remaining = valid_images[~valid_images["sample_id"].isin(processed_ids)]
    logger.info("Images remaining to process: %d", len(remaining))

    if remaining.empty:
        logger.info("All images already processed!")
        return pd.read_csv(checkpoint_path)

    # Initialize ImageAnalyzer
    img_analyzer = ImageAnalyzer(gpu=gpu)

    # Process images with progress logging
    new_rows = []
    t0 = time.time()
    total = len(remaining)

    for idx, (_, row) in enumerate(remaining.iterrows()):
        sample_id = row["sample_id"]
        img_path = row["local_image_path"]

        try:
            features = img_analyzer.analyze(
                img_path, run_watermark_check=run_watermark
            )
            # Remove ocr_text from features (we don't store it in CSV)
            features.pop("ocr_text", None)
            features["sample_id"] = sample_id
            new_rows.append(features)
        except Exception as e:
            logger.warning(
                "[%d/%d] Failed on %s: %s", idx + 1, total, sample_id, e
            )
            # Store defaults for failed images
            features = ImageAnalyzer.default_features()
            features.pop("ocr_text", None)
            features["sample_id"] = sample_id
            new_rows.append(features)

        # Progress logging
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate if rate > 0 else 0
            logger.info(
                "[%d/%d] %.1f img/sec, ETA: %.0fs",
                idx + 1, total, rate, eta,
            )

        # Checkpoint save
        if (idx + 1) % checkpoint_every == 0:
            _save_image_checkpoint(
                existing_rows + new_rows, checkpoint_path
            )
            logger.info("Checkpoint saved at %d images", idx + 1)

    # Final save
    all_rows = existing_rows + new_rows
    _save_image_checkpoint(all_rows, checkpoint_path)

    elapsed = time.time() - t0
    logger.info(
        "Image feature extraction complete: %d images in %.1f seconds",
        len(new_rows), elapsed,
    )

    return pd.read_csv(checkpoint_path)


def _save_image_checkpoint(rows: list, path: str):
    """Save image features checkpoint to CSV."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Merge and Compute Vision Score
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_score(
    text_features: pd.DataFrame,
    image_features: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """
    Merge text and image features, compute vision_score for every sample.

    Args:
        text_features: DataFrame from extract_text_features().
        image_features: DataFrame from extract_image_features().
        output_path: Path to save final vision_features.csv.

    Returns:
        Complete vision features DataFrame with one row per sample.
    """
    logger.info("Merging text features (%d) with image features (%d)...",
                len(text_features), len(image_features))

    # Start with text features (covers ALL samples)
    merged = text_features.copy()

    # Add has_image flag
    image_sample_ids = set(image_features["sample_id"].tolist()) if len(image_features) > 0 else set()
    merged["has_image"] = merged["sample_id"].isin(image_sample_ids).astype(int)

    # Add image feature columns with defaults
    image_cols = [
        "ocr_confidence", "tiny_text_count", "tiny_text_ratio",
        "footer_text_density", "footer_keyword_count",
        "hidden_text_detected", "hidden_text_count",
        "watermark_detected", "text_region_count", "spatial_spread",
    ]

    for col in image_cols:
        merged[col] = 0.0  # Default for text-only samples

    # Overwrite image columns for samples that have images
    if len(image_features) > 0 and "sample_id" in image_features.columns:
        img_indexed = image_features.set_index("sample_id")
        for col in image_cols:
            if col in img_indexed.columns:
                merged.loc[merged["sample_id"].isin(image_sample_ids), col] = (
                    merged.loc[merged["sample_id"].isin(image_sample_ids), "sample_id"]
                    .map(img_indexed[col])
                    .values
                )

    # Compute vision_score for every sample
    agent = VisionAgent.__new__(VisionAgent)  # No OCR reader needed
    agent.image_analyzer = None
    agent.text_analyzer = None

    logger.info("Computing vision_score for %d samples...", len(merged))
    vision_scores = merged.apply(
        lambda row: agent.compute_vision_score(row.to_dict()), axis=1
    )
    merged["vision_score"] = vision_scores

    # Validate
    assert merged["vision_score"].between(0, 1).all(), \
        "Vision scores must be in [0, 1]"
    assert not merged["sample_id"].duplicated().any(), \
        "Duplicate sample_ids found!"

    # Save
    merged.to_csv(output_path, index=False)
    logger.info("Vision features saved to %s (%d samples, %d columns)",
                output_path, len(merged), len(merged.columns))

    # Print summary stats
    print("\n" + "=" * 70)
    print("  VISION FEATURES SUMMARY")
    print("=" * 70)
    print(f"  Total samples:           {len(merged):,}")
    print(f"  Samples with images:     {merged['has_image'].sum():,}")
    print(f"  Samples text-only:       {(~merged['has_image'].astype(bool)).sum():,}")
    print(f"  Columns:                 {list(merged.columns)}")
    print(f"\n  Vision Score Stats:")
    print(f"    Mean:   {merged['vision_score'].mean():.4f}")
    print(f"    Median: {merged['vision_score'].median():.4f}")
    print(f"    Std:    {merged['vision_score'].std():.4f}")
    print(f"    Min:    {merged['vision_score'].min():.4f}")
    print(f"    Max:    {merged['vision_score'].max():.4f}")
    print(f"    >0:     {(merged['vision_score'] > 0).sum():,} samples")
    print(f"\n  Keyword Density Stats:")
    print(f"    >0:     {(merged['keyword_density'] > 0).sum():,} samples")
    print(f"    Mean (where >0): {merged.loc[merged['keyword_density'] > 0, 'keyword_density'].mean():.4f}")
    print("=" * 70)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_extraction(
    project_root: str,
    gpu: bool = True,
    text_col: str = "text",
    force_rerun_text: bool = False,
    force_rerun_images: bool = False,
    run_watermark: bool = False,
    checkpoint_every: int = 200,
) -> pd.DataFrame:
    """
    Run the complete vision feature extraction pipeline.

    This is the main entry point for Colab notebooks.

    Args:
        project_root: Path to PIID_PROJECT root (e.g., '/content/drive/MyDrive/PIID_PROJECT').
        gpu: Use GPU for EasyOCR.
        text_col: Text column name in benchmark_text.csv.
        force_rerun_text: Force re-extraction of text features.
        force_rerun_images: Force re-extraction of image features.
        run_watermark: Run watermark detection (slower).
        checkpoint_every: Save image checkpoint every N images.

    Returns:
        Complete vision features DataFrame.
    """
    paths = _resolve_paths(project_root)

    print("\n" + "=" * 70)
    print("  VISION BATCH RUNNER — Phase 3")
    print("=" * 70)
    print(f"  Project root:    {project_root}")
    print(f"  Text column:     {text_col}")
    print(f"  GPU:             {gpu}")
    print(f"  Watermark check: {run_watermark}")
    print("=" * 70 + "\n")

    t_start = time.time()

    # Step 1: Text features (all samples)
    print("\n>>> STEP 1: Text Feature Extraction (all samples)")
    print("-" * 50)
    text_features = extract_text_features(
        benchmark_text_path=str(paths["benchmark_text"]),
        output_path=str(paths["text_checkpoint"]),
        text_col=text_col,
        force_rerun=force_rerun_text,
    )

    # Step 2: Image features (image samples only)
    print("\n>>> STEP 2: Image Feature Extraction (image samples)")
    print("-" * 50)
    image_features = extract_image_features(
        metadata_path=str(paths["metadata"]),
        images_dir=str(paths["images_dir"]),
        checkpoint_path=str(paths["image_checkpoint"]),
        gpu=gpu,
        checkpoint_every=checkpoint_every,
        force_rerun=force_rerun_images,
        run_watermark=run_watermark,
    )

    # Step 3: Merge and compute vision scores
    print("\n>>> STEP 3: Merge Features & Compute Vision Scores")
    print("-" * 50)
    final = merge_and_score(
        text_features=text_features,
        image_features=image_features,
        output_path=str(paths["output_csv"]),
    )

    # Save run log
    elapsed = time.time() - t_start
    run_log = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_samples": len(final),
        "image_samples": int(final["has_image"].sum()),
        "elapsed_seconds": round(elapsed, 1),
        "text_col_used": text_col,
        "gpu_used": gpu,
        "watermark_check": run_watermark,
        "output_path": str(paths["output_csv"]),
        "vision_agent_version": VisionAgent.VERSION,
    }
    with open(str(paths["run_log"]), "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   Output: {paths['output_csv']}")

    return final


# ─────────────────────────────────────────────────────────────────────────────
# CLI / direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run vision feature extraction on PIID benchmark dataset"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="/content/drive/MyDrive/PIID_PROJECT",
        help="Path to PIID_PROJECT root directory",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--text-col", type=str, default="text",
                        help="Text column name")
    parser.add_argument("--force-text", action="store_true",
                        help="Force re-run text features")
    parser.add_argument("--force-images", action="store_true",
                        help="Force re-run image features")
    parser.add_argument("--watermark", action="store_true",
                        help="Enable watermark detection (slower)")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                        help="Checkpoint frequency for images")

    args = parser.parse_args()

    run_batch_extraction(
        project_root=args.project_root,
        gpu=not args.no_gpu,
        text_col=args.text_col,
        force_rerun_text=args.force_text,
        force_rerun_images=args.force_images,
        run_watermark=args.watermark,
        checkpoint_every=args.checkpoint_every,
    )
