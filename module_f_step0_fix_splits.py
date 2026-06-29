"""
Module F — Step 0: Fix Group Leakage in Train/Val/Test Splits
=============================================================
Loads the frozen parquet splits, extracts group keys from sample_ids,
applies GroupShuffleSplit to produce leak-free 80/10/10 splits, and
writes corrected_splits.csv to EXPERIMENT_CACHE/.

IMMUTABILITY: This script NEVER writes to FINAL_BENCHMARK_DATASET/.
All outputs go to EXPERIMENT_CACHE/.

Usage (Colab):
    !python module_f_step0_fix_splits.py

Usage (Local):
    uv run --with pandas --with pyarrow --with scikit-learn module_f_step0_fix_splits.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ── Path Configuration ──────────────────────────────────────────────
# Auto-detect environment: Colab (Drive) vs Local (Windows/Linux)
if os.path.exists("/content/drive/MyDrive/PAS/FINAL_BENCHMARK_DATASET"):
    # Google Colab with Drive mounted
    DATASET_DIR = "/content/drive/MyDrive/PAS/FINAL_BENCHMARK_DATASET"
    CACHE_DIR = "/content/drive/MyDrive/PAS/EXPERIMENT_CACHE"
else:
    # Local development
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(SCRIPT_DIR, "DATASETS", "FINAL_BENCHMARK_DATASET")
    CACHE_DIR = os.path.join(SCRIPT_DIR, "DATASETS", "EXPERIMENT_CACHE")

os.makedirs(CACHE_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE_FROM_REMAINDER = 0.1111  # 10/90 ≈ 0.1111 → gives 10% of total

# ── Step 1: Load Frozen Parquets ────────────────────────────────────
print("=" * 70)
print("MODULE F — STEP 0: FIX GROUP LEAKAGE")
print("=" * 70)

print(f"\n[1/6] Loading frozen parquet files from:\n      {DATASET_DIR}")

train_df = pd.read_parquet(os.path.join(DATASET_DIR, "train.parquet"))
val_df = pd.read_parquet(os.path.join(DATASET_DIR, "validation.parquet"))
test_df = pd.read_parquet(os.path.join(DATASET_DIR, "test.parquet"))

combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"      Combined: {len(combined):,} rows")
print(f"      Benign:   {(combined['label'] == 'benign').sum():,}")
print(f"      Malicious:{(combined['label'] == 'malicious').sum():,}")

assert len(combined) == 289874, f"Expected 289,874 rows, got {len(combined):,}"

# ── Step 2: Extract Group Keys ──────────────────────────────────────
print(f"\n[2/6] Extracting group keys from sample_id...")

def extract_group_key(sample_id: str) -> str:
    """Extract numeric suffix as group key: 'benign_00000004' → '00000004'"""
    parts = sample_id.split("_", 1)
    return parts[1] if len(parts) == 2 else sample_id

combined["group_key"] = combined["sample_id"].apply(extract_group_key)
n_groups = combined["group_key"].nunique()
print(f"      Unique group keys: {n_groups:,}")

# Verify: each group key should have exactly 2 samples (1 benign + 1 malicious)
group_sizes = combined.groupby("group_key").size()
print(f"      Min group size: {group_sizes.min()}")
print(f"      Max group size: {group_sizes.max()}")
print(f"      Mean group size: {group_sizes.mean():.2f}")

# ── Step 3: Show Current Leakage (Before Fix) ──────────────────────
print(f"\n[3/6] Measuring current group leakage (before fix)...")

# Reconstruct the original split column
combined["original_split"] = "unknown"
combined.loc[combined["sample_id"].isin(train_df["sample_id"]), "original_split"] = "train"
combined.loc[combined["sample_id"].isin(val_df["sample_id"]), "original_split"] = "val"
combined.loc[combined["sample_id"].isin(test_df["sample_id"]), "original_split"] = "test"

original_leakage = combined.groupby("group_key")["original_split"].nunique()
leaked_groups_before = (original_leakage > 1).sum()
print(f"      Groups with leakage (before): {leaked_groups_before:,} / {n_groups:,} "
      f"({leaked_groups_before / n_groups * 100:.2f}%)")

# ── Step 4: Apply GroupShuffleSplit ─────────────────────────────────
print(f"\n[4/6] Applying GroupShuffleSplit (random_state={RANDOM_STATE})...")

groups = combined["group_key"].values
labels = combined["label"].values

# First split: separate test set (10%)
gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
remainder_idx, test_idx = next(gss_test.split(combined, labels, groups))

# Second split: separate val from remainder (10% of total ≈ 11.11% of remainder)
remainder_groups = groups[remainder_idx]
remainder_labels = labels[remainder_idx]
gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE_FROM_REMAINDER, random_state=RANDOM_STATE)
train_idx_in_rem, val_idx_in_rem = next(gss_val.split(
    combined.iloc[remainder_idx], remainder_labels, remainder_groups
))

# Map back to original indices
train_idx = remainder_idx[train_idx_in_rem]
val_idx = remainder_idx[val_idx_in_rem]

# Assign corrected splits
combined["corrected_split"] = ""
combined.iloc[train_idx, combined.columns.get_loc("corrected_split")] = "train"
combined.iloc[val_idx, combined.columns.get_loc("corrected_split")] = "val"
combined.iloc[test_idx, combined.columns.get_loc("corrected_split")] = "test"

assert (combined["corrected_split"] == "").sum() == 0, "Some rows were not assigned a split!"

print(f"      Train: {len(train_idx):,} rows")
print(f"      Val:   {len(val_idx):,} rows")
print(f"      Test:  {len(test_idx):,} rows")

# ── Step 5: Verify Zero Leakage ────────────────────────────────────
print(f"\n[5/6] Running verification checks...")

checks = {}

# Check 1: Zero group leakage
corrected_leakage = combined.groupby("group_key")["corrected_split"].nunique()
leaked_groups_after = (corrected_leakage > 1).sum()
checks["group_leakage_count"] = leaked_groups_after
print(f"      [CHECK] Group leakage after fix: {leaked_groups_after} "
      f"({'[PASS]' if leaked_groups_after == 0 else '[FAIL]'})")

# Check 2: Total row count
checks["total_rows"] = len(combined)
print(f"      [CHECK] Total rows: {len(combined):,} "
      f"({'[PASS]' if len(combined) == 289874 else '[FAIL]'})")

# Check 3: No duplicate sample_ids
n_unique = combined["sample_id"].nunique()
checks["unique_sample_ids"] = n_unique
print(f"      [CHECK] Unique sample_ids: {n_unique:,} "
      f"({'[PASS]' if n_unique == 289874 else '[FAIL]'})")

# Check 4: Class balance per split
print(f"      [CHECK] Class balance per split:")
for split_name in ["train", "val", "test"]:
    split_data = combined[combined["corrected_split"] == split_name]
    n_benign = (split_data["label"] == "benign").sum()
    n_malicious = (split_data["label"] == "malicious").sum()
    ratio = n_benign / len(split_data) if len(split_data) > 0 else 0
    balance_ok = abs(ratio - 0.50) < 0.01
    checks[f"{split_name}_benign_ratio"] = round(ratio, 4)
    checks[f"{split_name}_total"] = len(split_data)
    checks[f"{split_name}_benign"] = n_benign
    checks[f"{split_name}_malicious"] = n_malicious
    print(f"             {split_name:>5}: {len(split_data):>7,} total | "
          f"B={n_benign:>6,} M={n_malicious:>6,} | "
          f"ratio={ratio:.4f} {'[PASS]' if balance_ok else '[WARN]'}")

# Check 5: Split proportions
train_pct = len(train_idx) / len(combined) * 100
val_pct = len(val_idx) / len(combined) * 100
test_pct = len(test_idx) / len(combined) * 100
checks["train_pct"] = round(train_pct, 2)
checks["val_pct"] = round(val_pct, 2)
checks["test_pct"] = round(test_pct, 2)
print(f"      [CHECK] Split proportions: "
      f"train={train_pct:.2f}% val={val_pct:.2f}% test={test_pct:.2f}%")

# ── Step 6: Save Outputs ───────────────────────────────────────────
print(f"\n[6/6] Saving outputs to:\n      {CACHE_DIR}")

# Save corrected_splits.csv
splits_output = combined[["sample_id", "corrected_split"]].copy()
splits_path = os.path.join(CACHE_DIR, "corrected_splits.csv")
splits_output.to_csv(splits_path, index=False)
print(f"      [OK] corrected_splits.csv ({len(splits_output):,} rows)")

# Save verification report
report_rows = []
for key, value in checks.items():
    report_rows.append({"check_name": key, "value": value})
report_df = pd.DataFrame(report_rows)
report_path = os.path.join(CACHE_DIR, "split_verification_report.csv")
report_df.to_csv(report_path, index=False)
print(f"      [OK] split_verification_report.csv ({len(report_rows)} checks)")

# ── Summary ─────────────────────────────────────────────────────────
all_passed = (
    leaked_groups_after == 0
    and len(combined) == 289874
    and n_unique == 289874
)

print(f"\n{'=' * 70}")
if all_passed:
    print("[PASS] MODULE F STEP 0 COMPLETE -- Zero group leakage confirmed.")
else:
    print("[FAIL] MODULE F STEP 0 -- SOME CHECKS FAILED. Review output above.")
print(f"{'=' * 70}")

sys.exit(0 if all_passed else 1)
