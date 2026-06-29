"""
AGENTS/preprocessing_agent.py
==============================
Handles data loading, label encoding, tokenisation strategy, and
train/val/test split management for the Prompt Analysis Agent pipeline.

Key design decisions (documented for the paper):
  - Labels are string 'benign'/'malicious' in parquet → encoded to 0/1
  - Truncation strategy: first-128 + last-128 + mid-256 tokens (max 512)
    This is a research contribution — captures header, body, and footer
    injections with a fixed budget, beating naive head truncation.
  - Scaler and feature column order are NEVER fit on val/test (enforced).
  - Corrected splits from EXPERIMENT_CACHE are used (not original parquets)
    to avoid the GroupShuffle data leakage fixed by module_f_step0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Path resolution (Colab or Local) ─────────────────────────────────────────

_COLAB_BASE = "/content/drive/MyDrive/PAS"
_LOCAL_BASE = str(Path(__file__).resolve().parents[1])  # repo root

def _resolve(colab_rel: str, local_rel: str) -> str:
    if os.path.exists(_COLAB_BASE):
        return os.path.join(_COLAB_BASE, colab_rel)
    return os.path.join(_LOCAL_BASE, local_rel)

DATASET_DIR  = _resolve("FINAL_BENCHMARK_DATASET", "DATASETS/FINAL_BENCHMARK_DATASET")
CACHE_DIR    = _resolve("EXPERIMENT_CACHE",         "DATASETS/EXPERIMENT_CACHE")
MODELS_DIR   = _resolve("MODELS",                   "MODELS")
RESULTS_DIR  = _resolve("RESULTS",                  "RESULTS")

LABEL_MAP = {"benign": 0, "malicious": 1}
LABEL_INV = {0: "benign", 1: "malicious"}


# ── Truncation strategy ───────────────────────────────────────────────────────

def smart_truncate(
    text: str,
    tokenizer,
    max_tokens: int = 512,
    head_tokens: int = 128,
    tail_tokens: int = 128,
) -> str:
    """
    First-128 + Last-128 + Mid-256 truncation strategy.

    For documents ≤ 512 tokens: returns as-is.
    For documents > 512 tokens: concatenates head, middle, and tail
    token slices.  This is better than naive head-truncation because
    prompt injections commonly appear in footers (attack_location=footer).

    Paper note: this strategy is compared against naive head-truncation
    in the ablation study (Notebook 08).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    mid_tokens = max_tokens - head_tokens - tail_tokens  # 256

    head = tokens[:head_tokens]
    tail = tokens[-tail_tokens:]
    mid_start = len(tokens) // 2 - mid_tokens // 2
    mid_start = max(head_tokens, mid_start)
    mid = tokens[mid_start: mid_start + mid_tokens]

    merged = head + mid + tail
    return tokenizer.decode(merged, skip_special_tokens=True)


# ── PreprocessingAgent ────────────────────────────────────────────────────────

class PreprocessingAgent:
    """
    Loads corrected splits, validates them, and prepares text + label arrays
    ready for RoBERTa training.

    Usage:
        agent = PreprocessingAgent()
        train_df, val_df, test_df = agent.load_splits()
        texts, labels = agent.prepare_for_roberta(train_df, tokenizer)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.dataset_dir = Path(DATASET_DIR)
        self._splits: Optional[Dict[str, pd.DataFrame]] = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads corrected_splits.csv (produced by module_f_step0_fix_splits.py).
        Falls back to original parquets if corrected file not found.

        Returns (train_df, val_df, test_df) with integer labels added.
        """
        corrected_path = self.cache_dir / "corrected_splits.csv"

        if corrected_path.exists():
            print(f"[PreprocessingAgent] Loading corrected splits from:\n  {corrected_path}")
            df = pd.read_csv(corrected_path)
        else:
            print("[PreprocessingAgent] WARNING: corrected_splits.csv not found. "
                  "Loading original parquets (may have group leakage).")
            train = pd.read_parquet(self.dataset_dir / "train.parquet")
            val   = pd.read_parquet(self.dataset_dir / "validation.parquet")
            test  = pd.read_parquet(self.dataset_dir / "test.parquet")
            for split_name, sdf in [("train", train), ("val", val), ("test", test)]:
                sdf["split"] = split_name
            df = pd.concat([train, val, test], ignore_index=True)

        # Encode labels
        df["label_int"] = df["label"].map(LABEL_MAP)
        if df["label_int"].isna().any():
            bad = df[df["label_int"].isna()]["label"].unique()
            raise ValueError(f"Unknown label values found: {bad}. "
                             f"Expected: {list(LABEL_MAP.keys())}")

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df   = df[df["split"] == "val"].reset_index(drop=True)
        test_df  = df[df["split"] == "test"].reset_index(drop=True)

        self._validate_splits(train_df, val_df, test_df)
        self._splits = {"train": train_df, "val": val_df, "test": test_df}

        print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        print(f"  Malicious %: train={train_df['label_int'].mean():.1%} "
              f"| val={val_df['label_int'].mean():.1%} "
              f"| test={test_df['label_int'].mean():.1%}")

        return train_df, val_df, test_df

    @staticmethod
    def _validate_splits(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Assert no sample_id appears in more than one split."""
        train_ids = set(train_df["sample_id"])
        val_ids   = set(val_df["sample_id"])
        test_ids  = set(test_df["sample_id"])

        tv = train_ids & val_ids
        tt = train_ids & test_ids
        vt = val_ids   & test_ids

        leaks = []
        if tv: leaks.append(f"train∩val={len(tv)}")
        if tt: leaks.append(f"train∩test={len(tt)}")
        if vt: leaks.append(f"val∩test={len(vt)}")

        if leaks:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED — overlapping sample_ids: {', '.join(leaks)}. "
                "Run module_f_step0_fix_splits.py first."
            )
        print("  [✓] No data leakage between splits.")

    # ── Text preparation ──────────────────────────────────────────────────────

    def get_text_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns the best available text column.
        Preference: 'ocr_text' (from cache) > 'text' > 'prompt'
        """
        for col in ["ocr_text", "text", "prompt"]:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"  Using text column: '{col}' ({non_null:,}/{len(df):,} non-null)")
                return df[col].fillna("")
        raise KeyError("No text column found. Expected one of: ocr_text, text, prompt")

    def prepare_for_roberta(
        self,
        df: pd.DataFrame,
        tokenizer=None,
        use_smart_truncation: bool = True,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Returns (texts, labels) ready for HuggingFace Dataset or DataLoader.

        If tokenizer is provided and use_smart_truncation=True, applies
        first-128 + last-128 + mid-256 truncation for long documents.
        """
        texts = self.get_text_column(df).tolist()
        labels = df["label_int"].values

        if tokenizer is not None and use_smart_truncation:
            print("  Applying smart truncation (head128+mid256+tail128)...")
            texts = [smart_truncate(t, tokenizer) for t in texts]

        return texts, labels

    # ── Class weights for imbalanced data ─────────────────────────────────────

    @staticmethod
    def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
        """
        Inverse-frequency class weights for CrossEntropy loss.
        Paper note: used when benign:malicious ratio > 3:1.
        """
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array([0, 1])
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        cw = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"  Class weights: benign={cw[0]:.3f}, malicious={cw[1]:.3f}")
        return cw

    # ── Metadata helpers ──────────────────────────────────────────────────────

    @staticmethod
    def get_attack_families(df: pd.DataFrame) -> pd.Series:
        """Returns attack_family column if present, else 'unknown'."""
        if "attack_family" in df.columns:
            return df["attack_family"].fillna("unknown")
        return pd.Series(["unknown"] * len(df), index=df.index)
