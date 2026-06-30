"""
AGENTS/ocr_adapter.py
=====================
Unified OCR interface used by module_f_step1_ocr_cache.py and VisionAgent.

Provides:
  OCREngineFactory.create(engine, languages, gpu) → OCREngine
  OCREngine.run(image_path)   → OCRResult(text, confidence, boxes, duration_ms)
  OCREngine.run_array(np_img) → OCRResult

Design decisions:
  - Factory pattern allows future swap to PaddleOCR / Tesseract without
    touching downstream code.
  - All outputs are typed dataclasses to prevent silent field mismatches.
  - Confidence is always normalised to [0, 1] regardless of backend.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class TextBox:
    """One detected text region from OCR."""
    text: str
    confidence: float          # [0, 1]
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) in pixels


@dataclass
class OCRResult:
    """Aggregated OCR output for a single document image."""
    text: str                         # Full concatenated text
    confidence: float                 # Mean confidence across all boxes
    boxes: List[TextBox] = field(default_factory=list)
    duration_ms: float = 0.0
    engine: str = "easyocr"
    error: Optional[str] = None       # Set if OCR failed gracefully

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


# ── EasyOCR Backend ──────────────────────────────────────────────────────────


class EasyOCREngine:
    """
    Wraps EasyOCR reader.  Lazily initialised on first call so that
    import of ocr_adapter does not force GPU/model initialisation.
    """

    def __init__(self, languages: List[str], gpu: bool = False):
        self.languages = languages
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr  # local import — allows module import without easyocr installed
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def run(self, image_path: str) -> OCRResult:
        """Run OCR on a file path."""
        path = Path(image_path)
        if not path.exists():
            return OCRResult(
                text="", confidence=0.0,
                error=f"File not found: {image_path}"
            )
        try:
            t0 = time.perf_counter()
            raw = self._get_reader().readtext(str(path), detail=1)
            duration_ms = (time.perf_counter() - t0) * 1000
            return self._parse_raw(raw, duration_ms)
        except Exception as exc:
            return OCRResult(
                text="", confidence=0.0,
                error=str(exc)
            )

    def run_array(self, img_array: np.ndarray) -> OCRResult:
        """Run OCR on an in-memory NumPy image (H×W×C, uint8)."""
        try:
            t0 = time.perf_counter()
            raw = self._get_reader().readtext(img_array, detail=1)
            duration_ms = (time.perf_counter() - t0) * 1000
            return self._parse_raw(raw, duration_ms)
        except Exception as exc:
            return OCRResult(
                text="", confidence=0.0,
                error=str(exc)
            )

    @staticmethod
    def _parse_raw(raw: list, duration_ms: float) -> OCRResult:
        """
        Convert EasyOCR raw output to OCRResult.
        EasyOCR returns: [ ([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], text, conf), ... ]
        """
        boxes: List[TextBox] = []
        texts: List[str] = []
        confidences: List[float] = []

        for (quad, text, conf) in raw:
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            conf_clamped = max(0.0, min(1.0, float(conf)))
            boxes.append(TextBox(text=text, confidence=conf_clamped, bbox=bbox))
            texts.append(text)
            confidences.append(conf_clamped)

        full_text = " ".join(texts)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=mean_conf,
            boxes=boxes,
            duration_ms=duration_ms,
            engine="easyocr",
        )


# ── Factory ──────────────────────────────────────────────────────────────────


class OCREngineFactory:
    """
    Usage:
        engine = OCREngineFactory.create("easyocr", languages=["en"], gpu=True)
        result = engine.run("/path/to/image.png")
        print(result.text, result.confidence)
    """

    _BACKENDS = {"easyocr": EasyOCREngine}

    @classmethod
    def create(
        cls,
        engine: str = "easyocr",
        languages: Optional[List[str]] = None,
        gpu: bool = False,
    ) -> EasyOCREngine:
        if languages is None:
            languages = ["en"]
        backend_cls = cls._BACKENDS.get(engine)
        if backend_cls is None:
            raise ValueError(
                f"Unknown OCR engine: '{engine}'. "
                f"Available: {list(cls._BACKENDS.keys())}"
            )
        return backend_cls(languages=languages, gpu=gpu)
