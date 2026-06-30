"""
AGENTS/vision_agent.py
=======================
Rule-based multimodal vision analysis agent.

Extracts 7 interpretable visual features from document images
to detect prompt injection attacks that survive or hide from text extraction.

Feature design rationale (for paper Section 4.3):
  1. ocr_confidence    — Low avg confidence = distorted/unusual text
  2. tiny_text_count   — Text boxes < 1% image height = hidden micro-text
  3. footer_density    — High text density in bottom 15% = footer injection
  4. watermark_score   — Text in low-opacity layer not in normal-contrast OCR
  5. hidden_text_score — Near-white text on white background
  6. keyword_density   — Injection keyword count / total words
  7. vision_score      — Weighted combination of above → [0, 1]

Why not CNN?  Rule-based features are:
  (a) Interpretable — coefficients explainable to regulators
  (b) Fast — microseconds vs 100ms+ for neural inference
  (c) Sufficient — attacks are OCR-visible by design (see dataset construction)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

warnings.filterwarnings("ignore")

# Injection keywords (OWASP LLM01 attack surface vocabulary)
INJECTION_KEYWORDS = [
    "ignore", "disregard", "forget", "override", "bypass", "pretend",
    "you are now", "act as", "roleplay", "system prompt", "system message",
    "previous instructions", "instruction override", "jailbreak",
    "ignore previous", "disregard previous", "new instructions",
    "ignore all", "ignore the above", "from now on",
    "reveal", "extract", "exfiltrate", "print your", "show me your",
    "what is your", "tell me your", "confidential", "internal",
    "claim manipulation", "approve claim", "deny claim",
    "inject", "payload", "malicious",
]


# ── VisionFeatures Dataclass ──────────────────────────────────────────────────

@dataclass
class VisionFeatures:
    """All extracted visual features for one document image."""
    sample_id: str
    image_path: str

    # Raw features
    ocr_confidence: float       # [0, 1] mean confidence of all OCR boxes
    tiny_text_count: int        # # boxes with height < 1% of image height
    footer_text_density: float  # # text boxes in bottom-15% / total boxes
    watermark_score: float      # [0, 1] new text found at high contrast
    hidden_text_score: float    # [0, 1] near-white text detection score
    keyword_density: float      # injection keyword count / total word count
    total_boxes: int            # total OCR text boxes detected
    total_words: int            # total word count from OCR

    # Derived
    vision_score: float         # weighted combination [0, 1]
    error: Optional[str] = None # set if processing failed

    def to_dict(self) -> dict:
        return asdict(self)


# ── Weight configuration (documented for paper) ───────────────────────────────

FEATURE_WEIGHTS = {
    "keyword_density":    0.35,   # strongest signal — direct semantic attack
    "hidden_text_score":  0.25,   # high severity — invisible content
    "watermark_score":    0.15,   # medium — low-opacity hidden text
    "footer_density":     0.10,   # medium — common injection location
    "tiny_text_count":    0.10,   # normalised — micro-font evasion
    "ocr_confidence":     0.05,   # inverted — low confidence = suspicious
}

assert abs(sum(FEATURE_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


# ── VisionAgent ───────────────────────────────────────────────────────────────

class VisionAgent:
    """
    Extracts visual features from a document image.

    Usage:
        agent = VisionAgent(gpu=True)
        features = agent.extract("path/to/image.png", sample_id="mal_00001")
        print(features.vision_score, features.keyword_density)

    Batch usage (for Notebook 03):
        results = agent.extract_batch(image_paths, sample_ids)
    """

    def __init__(self, gpu: bool = False, ocr_languages: List[str] = None):
        if ocr_languages is None:
            ocr_languages = ["en"]
        self.gpu = gpu
        self.ocr_languages = ocr_languages
        self._ocr_engine = None
        self._cv2 = None

    def _get_ocr(self):
        if self._ocr_engine is None:
            from ocr_adapter import OCREngineFactory
            self._ocr_engine = OCREngineFactory.create(
                "easyocr", languages=self.ocr_languages, gpu=self.gpu
            )
        return self._ocr_engine

    def _get_cv2(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2

    # ── Core extraction ───────────────────────────────────────────────────────

    def extract(self, image_path: str, sample_id: str = "") -> VisionFeatures:
        """Full feature extraction pipeline for one image."""
        path = Path(image_path)
        if not path.exists():
            return self._error_features(sample_id, image_path,
                                        f"File not found: {image_path}")
        try:
            cv2 = self._get_cv2()
            img = cv2.imread(str(path))
            if img is None:
                return self._error_features(sample_id, image_path, "cv2 failed to read image")

            img_h, img_w = img.shape[:2]
            ocr = self._get_ocr()

            # Normal OCR run
            normal_result = ocr.run(image_path)
            boxes = normal_result.boxes

            # Feature 1: OCR confidence
            ocr_confidence = normal_result.confidence

            # Feature 2: tiny text count
            tiny_text_count = self._count_tiny_text(boxes, img_h)

            # Feature 3: footer text density
            footer_density = self._footer_text_density(boxes, img_h)

            # Feature 4: watermark score
            watermark_score = self._watermark_score(img, ocr, normal_result)

            # Feature 5: hidden text score
            hidden_text_score = self._hidden_text_score(img, ocr)

            # Feature 6: keyword density
            keyword_density, total_words = self._keyword_density(normal_result.text)

            total_boxes = len(boxes)

            vision_score = self._compute_vision_score(
                ocr_confidence=ocr_confidence,
                tiny_text_count=tiny_text_count,
                footer_density=footer_density,
                watermark_score=watermark_score,
                hidden_text_score=hidden_text_score,
                keyword_density=keyword_density,
                total_boxes=total_boxes,
            )

            return VisionFeatures(
                sample_id=sample_id,
                image_path=image_path,
                ocr_confidence=ocr_confidence,
                tiny_text_count=tiny_text_count,
                footer_text_density=footer_density,
                watermark_score=watermark_score,
                hidden_text_score=hidden_text_score,
                keyword_density=keyword_density,
                total_boxes=total_boxes,
                total_words=total_words,
                vision_score=vision_score,
            )

        except Exception as exc:
            return self._error_features(sample_id, image_path, str(exc))

    # ── Individual feature methods ────────────────────────────────────────────

    @staticmethod
    def _count_tiny_text(boxes, img_h: int) -> int:
        """
        Count text boxes whose bounding box height < 1% of image height.
        Normalised by log(1 + count) for scale invariance across image sizes.
        Raw count returned here; normalisation happens in vision_score.
        """
        threshold = img_h * 0.01
        return sum(1 for b in boxes if (b.bbox[3] - b.bbox[1]) < threshold)

    @staticmethod
    def _footer_text_density(boxes, img_h: int) -> float:
        """
        Fraction of text boxes whose centre-y falls in the bottom 15% of image.
        Returns [0, 1].
        """
        if not boxes:
            return 0.0
        footer_start = img_h * 0.85
        footer_boxes = sum(
            1 for b in boxes
            if ((b.bbox[1] + b.bbox[3]) / 2) >= footer_start
        )
        return footer_boxes / len(boxes)

    def _watermark_score(self, img, ocr, normal_result) -> float:
        """
        Run OCR on a high-contrast version of the image.
        New words appearing only in high-contrast OCR are potential watermarks.
        Score = fraction of high-contrast words not found in normal OCR.
        """
        try:
            cv2 = self._get_cv2()
            # Increase contrast: alpha=2.5, beta=-100
            high_contrast = cv2.convertScaleAbs(img, alpha=2.5, beta=-100)
            hc_result = ocr.run_array(high_contrast)

            normal_words = set(normal_result.text.lower().split())
            hc_words = set(hc_result.text.lower().split())
            new_words = hc_words - normal_words

            if not hc_words:
                return 0.0
            return min(1.0, len(new_words) / max(1, len(hc_words)))
        except Exception:
            return 0.0

    def _hidden_text_score(self, img, ocr) -> float:
        """
        Detect near-white text on white background.
        Strategy: threshold to isolate near-white pixels, run OCR on mask.
        Score = 1.0 if any text found in near-white regions, else 0.0.
        """
        try:
            cv2 = self._get_cv2()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Near-white: pixel values 230-255
            _, near_white_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
            # Invert — isolate near-white regions
            masked = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(near_white_mask))
            # Enhance contrast in masked region
            enhanced = cv2.convertScaleAbs(masked, alpha=5.0, beta=0)
            result = ocr.run_array(enhanced)
            if result.is_empty or result.error:
                return 0.0
            # Non-empty OCR in near-white region → hidden text detected
            return 1.0 if len(result.text.strip()) > 3 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _keyword_density(text: str):
        """
        Count injection keywords / total words.
        Returns (density_float, total_word_count).
        """
        if not text:
            return 0.0, 0
        words = text.lower().split()
        total = len(words)
        if total == 0:
            return 0.0, 0
        text_lower = text.lower()
        count = sum(1 for kw in INJECTION_KEYWORDS if kw in text_lower)
        density = min(1.0, count / max(1, total / 10))  # normalise per 10 words
        return density, total

    @staticmethod
    def _compute_vision_score(
        ocr_confidence: float,
        tiny_text_count: int,
        footer_density: float,
        watermark_score: float,
        hidden_text_score: float,
        keyword_density: float,
        total_boxes: int,
    ) -> float:
        """
        Weighted combination of features → vision_score ∈ [0, 1].

        Note: ocr_confidence is INVERTED (low conf = high suspicion).
        tiny_text_count is log-normalised and capped at 1.
        """
        inv_confidence = 1.0 - ocr_confidence
        norm_tiny = min(1.0, np.log1p(tiny_text_count) / np.log1p(10))

        score = (
            FEATURE_WEIGHTS["keyword_density"]    * keyword_density    +
            FEATURE_WEIGHTS["hidden_text_score"]   * hidden_text_score  +
            FEATURE_WEIGHTS["watermark_score"]     * watermark_score    +
            FEATURE_WEIGHTS["footer_density"]      * footer_density     +
            FEATURE_WEIGHTS["tiny_text_count"]     * norm_tiny          +
            FEATURE_WEIGHTS["ocr_confidence"]      * inv_confidence
        )
        return float(np.clip(score, 0.0, 1.0))

    # ── Batch extraction ──────────────────────────────────────────────────────

    def extract_batch(
        self,
        image_paths: List[str],
        sample_ids: List[str],
        verbose: bool = True,
    ) -> List[VisionFeatures]:
        """
        Process multiple images.  Shows progress bar in Colab-compatible way.
        """
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(zip(image_paths, sample_ids), total=len(image_paths),
                            desc="VisionAgent")
        except ImportError:
            iterator = zip(image_paths, sample_ids)

        results = []
        for path, sid in iterator:
            features = self.extract(path, sid)
            results.append(features)
            if verbose and features.error:
                print(f"  [WARN] {sid}: {features.error}")
        return results

    # ── Error fallback ────────────────────────────────────────────────────────

    @staticmethod
    def _error_features(sample_id: str, image_path: str, error: str) -> VisionFeatures:
        """
        Returns a conservative high-risk feature set when processing fails.
        Fail-safe design: unknown = elevated risk.
        """
        return VisionFeatures(
            sample_id=sample_id, image_path=image_path,
            ocr_confidence=0.0, tiny_text_count=0,
            footer_text_density=0.0, watermark_score=0.0,
            hidden_text_score=0.0, keyword_density=0.0,
            total_boxes=0, total_words=0,
            vision_score=0.5,   # elevated uncertainty score
            error=error,
        )
