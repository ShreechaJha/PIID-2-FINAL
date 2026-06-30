"""
AGENTS/risk_agent.py
====================
Risk Assessment Agent: fuses prompt + vision signals via Logistic Regression.

Why Logistic Regression (paper Section 4.4):
  1. Interpretable — feature coefficients reportable to regulators
  2. Fast — microsecond inference vs neural alternatives
  3. Well-calibrated probability output by default
  4. Explicitly separable from governance rules (defence in depth)

Calibration (research contribution):
  Platt scaling applied post-training ensures risk_score=0.8 truly means
  80% of such inputs are malicious.  This is required for meaningful
  governance rule thresholds.  Reported as calibration curve in paper.

Data leakage prevention (checked at training time):
  - StandardScaler fit on training data ONLY
  - Feature column order persisted to JSON (inference must match exactly)
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_COLAB_BASE = "/content/drive/MyDrive/PAS"
_LOCAL_BASE = str(Path(__file__).resolve().parents[1])

def _models_dir() -> Path:
    if os.path.exists(_COLAB_BASE):
        return Path(_COLAB_BASE) / "MODELS"
    return Path(_LOCAL_BASE) / "MODELS"


# ── RiskScore ─────────────────────────────────────────────────────────────────

@dataclass
class RiskScore:
    sample_id: str
    risk_score: float          # calibrated probability [0, 1]
    risk_level: str            # LOW / MEDIUM / HIGH / CRITICAL
    feature_values: dict       # for audit log
    duration_ms: float

    @property
    def is_high_risk(self) -> bool:
        return self.risk_score >= 0.65


# ── Risk thresholds ───────────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    "LOW":      (0.0,  0.35),
    "MEDIUM":   (0.35, 0.65),
    "HIGH":     (0.65, 0.80),
    "CRITICAL": (0.80, 1.01),
}

def _risk_level(score: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "CRITICAL"


# ── RiskAgent ─────────────────────────────────────────────────────────────────

class RiskAgent:
    """
    Loads a trained LogisticRegression + StandardScaler and scores documents.

    Usage (after Notebook 05):
        agent = RiskAgent()
        score = agent.score(features_dict, sample_id="test_001")
        print(score.risk_score, score.risk_level)
    """

    LR_FILENAME      = "logistic_regression.pkl"
    SCALER_FILENAME  = "scaler.pkl"
    COLUMNS_FILENAME = "feature_columns.json"

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else _models_dir()
        self._lr = None
        self._scaler = None
        self._columns: Optional[List[str]] = None

    def _load(self):
        if self._lr is not None:
            return
        lr_path  = self.models_dir / self.LR_FILENAME
        sc_path  = self.models_dir / self.SCALER_FILENAME
        col_path = self.models_dir / self.COLUMNS_FILENAME

        for p in [lr_path, sc_path, col_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Model file missing: {p}\n"
                    "Run Notebook 05 to train and save the risk model."
                )
        with open(lr_path, "rb") as f:
            self._lr = pickle.load(f)
        with open(sc_path, "rb") as f:
            self._scaler = pickle.load(f)
        with open(col_path) as f:
            self._columns = json.load(f)["columns"]

        print(f"[RiskAgent] Loaded model from: {self.models_dir}")
        print(f"[RiskAgent] Feature columns ({len(self._columns)}): {self._columns[:5]} ...")

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        features: Dict[str, float],
        sample_id: str = "",
    ) -> RiskScore:
        """
        Score a single document given a feature dictionary.

        features dict should contain keys matching feature_columns.json.
        Missing features are filled with 0.0 (conservative).
        """
        self._load()
        t0 = time.perf_counter()

        # Build feature vector in correct column order
        x = np.array([[features.get(col, 0.0) for col in self._columns]])
        x_scaled = self._scaler.transform(x)
        prob = self._lr.predict_proba(x_scaled)[0, 1]   # malicious probability
        prob = float(np.clip(prob, 0.0, 1.0))
        duration_ms = (time.perf_counter() - t0) * 1000

        return RiskScore(
            sample_id=sample_id,
            risk_score=prob,
            risk_level=_risk_level(prob),
            feature_values=features,
            duration_ms=duration_ms,
        )

    def score_batch(
        self,
        features_list: List[Dict[str, float]],
        sample_ids: Optional[List[str]] = None,
    ) -> List[RiskScore]:
        """Batch scoring — much faster than looping score()."""
        self._load()
        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(features_list))]

        t0 = time.perf_counter()
        X = np.array([
            [f.get(col, 0.0) for col in self._columns]
            for f in features_list
        ])
        X_scaled = self._scaler.transform(X)
        probs = self._lr.predict_proba(X_scaled)[:, 1].astype(float)
        total_ms = (time.perf_counter() - t0) * 1000

        return [
            RiskScore(
                sample_id=sid,
                risk_score=float(np.clip(p, 0.0, 1.0)),
                risk_level=_risk_level(float(p)),
                feature_values=f,
                duration_ms=total_ms / len(features_list),
            )
            for sid, p, f in zip(sample_ids, probs, features_list)
        ]

    # ── Feature importance (for paper) ────────────────────────────────────────

    def feature_importance(self) -> Dict[str, float]:
        """
        Returns feature → coefficient magnitude dict (sorted descending).
        Use this to populate Table 3 in the paper.
        """
        self._load()
        coefs = self._lr.coef_[0]
        importance = {col: float(abs(c)) for col, c in zip(self._columns, coefs)}
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ── Build feature dict (helper for DecisionAgent) ─────────────────────────

    @staticmethod
    def build_feature_dict(
        malicious_probability: float,
        vision_features: dict,
        severity: str = "medium",
        attack_type: str = "unknown",
    ) -> Dict[str, float]:
        """
        Assembles the feature dictionary expected by score().
        Maps categorical fields to numeric encodings documented in Notebook 04.

        Severity encoding (ordinal — has natural order):
            low=0, medium=1, high=2, critical=3
        """
        SEV_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        return {
            # From PromptAgent
            "malicious_probability": malicious_probability,

            # From VisionAgent
            "ocr_confidence":       vision_features.get("ocr_confidence", 0.5),
            "tiny_text_count":      vision_features.get("tiny_text_count", 0),
            "footer_text_density":  vision_features.get("footer_text_density", 0.0),
            "watermark_score":      vision_features.get("watermark_score", 0.0),
            "hidden_text_score":    vision_features.get("hidden_text_score", 0.0),
            "keyword_density":      vision_features.get("keyword_density", 0.0),
            "vision_score":         vision_features.get("vision_score", 0.0),

            # From metadata (encoded)
            "severity_enc": float(SEV_MAP.get(str(severity).lower(), 1)),
        }
