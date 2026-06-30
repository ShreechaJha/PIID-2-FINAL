"""
AGENTS/prompt_agent.py
=======================
RoBERTa-base binary classifier for prompt injection detection in text.

This agent is the core ML component.  After training (Notebook 02),
this module loads the saved model and provides batch inference.

Architecture choices (documented for paper Section 4.2):
  - RoBERTa-base: 125M params, T4-trainable, strong classification baseline
  - Sliding window inference for long documents (research contribution)
  - Outputs calibrated malicious_probability [0,1] — used by RiskAgent
  - No LLM (GPT/LLaMA) — reproducibility + interpretability requirement
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# Resolve MODELS dir
_COLAB_BASE = "/content/drive/MyDrive/PAS"
_LOCAL_BASE = str(Path(__file__).resolve().parents[1])

def _models_dir() -> Path:
    if os.path.exists(_COLAB_BASE):
        return Path(_COLAB_BASE) / "MODELS"
    return Path(_LOCAL_BASE) / "MODELS"


# ── PromptPrediction ──────────────────────────────────────────────────────────

@dataclass
class PromptPrediction:
    """Output for one text sample."""
    sample_id: str
    text_preview: str                 # first 80 chars
    malicious_probability: float      # [0, 1]
    predicted_label: int              # 0=benign, 1=malicious
    confidence: float                 # max(prob, 1-prob)
    predicted_attack_family: str      # 'unknown' unless multi-class
    inference_strategy: str           # 'direct' or 'sliding_window'
    duration_ms: float


# ── PromptAgent ───────────────────────────────────────────────────────────────

class PromptAgent:
    """
    Loads a fine-tuned RoBERTa-base model and runs binary classification.

    Usage (after training):
        agent = PromptAgent()
        pred = agent.predict("Ignore all previous instructions and ...")
        print(pred.malicious_probability)   # e.g. 0.97

    Batch:
        preds = agent.predict_batch(texts, sample_ids)
    """

    MODEL_DIR_NAME = "roberta_classifier"
    MAX_LENGTH = 512
    WINDOW_SIZE = 512
    WINDOW_STRIDE = 256   # 50% overlap

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        use_sliding_window: bool = True,
    ):
        self.model_dir = Path(model_dir) if model_dir else _models_dir() / self.MODEL_DIR_NAME
        self.threshold = threshold
        self.use_sliding_window = use_sliding_window
        self._model = None
        self._tokenizer = None
        self._device = None
        self._device_str = device

    def _load(self):
        """Lazy load model + tokenizer on first inference call."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not self.model_dir.exists():
            print(f"[PromptAgent] Local model not found at {self.model_dir}. Falling back to 'roberta-base' from Hugging Face Hub.")
            self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self._model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        else:
            if (self.model_dir / "best_strategyA").exists() and (self.model_dir / "best_strategyA" / "config.json").exists():
                self.model_dir = self.model_dir / "best_strategyA"
            print(f"[PromptAgent] Loading model from: {self.model_dir}")
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir)
            )
        if self._device_str:
            self._device = torch.device(self._device_str)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model.to(self._device)
        self._model.eval()
        print(f"[PromptAgent] Model loaded on {self._device}")

    # ── Single prediction ─────────────────────────────────────────────────────

    def predict(self, text: str, sample_id: str = "") -> PromptPrediction:
        """Predict malicious probability for a single text."""
        self._load()
        import torch

        t0 = time.perf_counter()
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.MAX_LENGTH - 2 or not self.use_sliding_window:
            prob = self._infer_direct(text)
            strategy = "direct"
        else:
            prob = self._infer_sliding_window(tokens)
            strategy = "sliding_window"

        duration_ms = (time.perf_counter() - t0) * 1000
        label = int(prob >= self.threshold)
        conf = max(prob, 1 - prob)

        return PromptPrediction(
            sample_id=sample_id,
            text_preview=text[:80].replace("\n", " "),
            malicious_probability=float(prob),
            predicted_label=label,
            confidence=float(conf),
            predicted_attack_family="unknown",
            inference_strategy=strategy,
            duration_ms=duration_ms,
        )

    def _infer_direct(self, text: str) -> float:
        """Run model on the full text (≤ 512 tokens)."""
        import torch
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding=True,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item()   # malicious class probability

    def _infer_sliding_window(self, tokens: list) -> float:
        """
        Sliding window for long documents.
        Max-pooling on malicious probability across windows.
        Research rationale: captures injections anywhere in the document.
        """
        import torch
        max_prob = 0.0
        inner = self.MAX_LENGTH - 2  # excluding [CLS] and [SEP]

        for start in range(0, len(tokens), self.WINDOW_STRIDE):
            chunk = tokens[start: start + inner]
            if not chunk:
                break
            chunk_text = self._tokenizer.decode(chunk, skip_special_tokens=True)
            prob = self._infer_direct(chunk_text)
            max_prob = max(max_prob, prob)
            if max_prob > 0.99:   # early exit for obvious cases
                break

        return max_prob

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[PromptPrediction]:
        """
        Efficient batch inference.  Uses DataLoader-style batching for
        direct inference.  Falls back to single inference for long texts.
        """
        self._load()
        import torch

        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(texts))]

        predictions = []
        try:
            from tqdm.auto import tqdm
            range_iter = tqdm(range(0, len(texts), batch_size), desc="PromptAgent")
        except ImportError:
            range_iter = range(0, len(texts), batch_size)

        for i in range_iter:
            batch_texts = texts[i: i + batch_size]
            batch_ids = sample_ids[i: i + batch_size]

            # Check if any text needs sliding window
            for text, sid in zip(batch_texts, batch_ids):
                pred = self.predict(text, sid)
                predictions.append(pred)

        return predictions

    def predictions_to_dataframe(self, predictions: List[PromptPrediction]):
        """Convert list of PromptPredictions to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([
            {
                "sample_id": p.sample_id,
                "malicious_probability": p.malicious_probability,
                "predicted_label": p.predicted_label,
                "confidence": p.confidence,
                "predicted_attack_family": p.predicted_attack_family,
                "inference_strategy": p.inference_strategy,
                "duration_ms": p.duration_ms,
            }
            for p in predictions
        ])
