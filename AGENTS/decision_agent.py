"""
AGENTS/decision_agent.py
=========================
Orchestrator: coordinates all agents end-to-end and produces the final
auditable decision with structured logging.

Execution sequence:
  1. PreprocessingAgent  — load & validate text
  2. VisionAgent         — extract visual features (if image provided)
  3. PromptAgent         — RoBERTa malicious_probability
  4. RiskAgent           — fuse features → risk_score
  5. GovernanceAgent     — apply policy rules → ALLOW/BLOCK/SANITIZE
  6. Audit log           — write JSON Lines entry to RESULTS/audit_log.jsonl

Fail-safe design:
  Any agent failure → elevated risk score, conservative decision.
  A security system must fail safe, not fail open.

Paper note (Section 4.6):
  The audit log implements the 'adverse action notice' pattern required
  in regulated industries — every automated decision is fully traceable.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add AGENTS/ to sys.path so agents can import each other
_AGENTS_DIR = Path(__file__).resolve().parent
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from ocr_adapter import OCREngineFactory
from vision_agent import VisionAgent, VisionFeatures
from prompt_agent import PromptAgent
from risk_agent import RiskAgent
from governance_agent import GovernanceAgent, GovernanceDecision

_COLAB_BASE = "/content/drive/MyDrive/PAS"
_LOCAL_BASE = str(Path(__file__).resolve().parents[1])

def _results_dir() -> Path:
    if os.path.exists(_COLAB_BASE):
        return Path(_COLAB_BASE) / "RESULTS"
    return Path(_LOCAL_BASE) / "RESULTS"

AGENT_VERSION = "1.0.0"


# ── FinalDecision ─────────────────────────────────────────────────────────────

@dataclass
class FinalDecision:
    """Complete output of the DecisionAgent pipeline."""
    # Identity
    audit_log_id: str
    sample_id: str
    timestamp: str

    # Agent outputs
    decision: str                    # ALLOW | BLOCK | SANITIZE
    risk_score: float
    risk_level: str
    prompt_score: float
    vision_score: float
    confidence: float

    # Governance
    governance_rule_triggered: Optional[str]
    reason: str
    sanitization_action: Optional[str]
    policy_ref: Optional[str]

    # Latency breakdown (ms)
    vision_ms: float
    prompt_ms: float
    risk_ms: float
    governance_ms: float
    total_ms: float

    # Errors (None if clean run)
    vision_error: Optional[str]
    prompt_error: Optional[str]

    # Agent versions for reproducibility
    agent_version: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_response_json(self) -> dict:
        """Slim response for the FastAPI endpoint."""
        return {
            "decision": self.decision,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "prompt_score": round(self.prompt_score, 4),
            "vision_score": round(self.vision_score, 4),
            "governance_rule_triggered": self.governance_rule_triggered,
            "reason": self.reason,
            "sanitization_action": self.sanitization_action,
            "policy_ref": self.policy_ref,
            "processing_time_ms": round(self.total_ms, 2),
            "audit_log_id": self.audit_log_id,
        }


# ── DecisionAgent ─────────────────────────────────────────────────────────────

class DecisionAgent:
    """
    Full pipeline orchestrator.

    Usage:
        agent = DecisionAgent()
        result = agent.process(image_path="doc.png", text="extracted text")
        print(result.decision, result.reason)

    The agent auto-detects what's available:
      - image_path only → run VisionAgent + OCR for text
      - text only       → run PromptAgent only, skip VisionAgent
      - both            → full multimodal pipeline
      - neither         → raise ValueError
    """

    def __init__(
        self,
        gpu: bool = False,
        audit_log_path: Optional[str] = None,
        models_dir: Optional[str] = None,
        rules_path: Optional[str] = None,
        prompt_threshold: float = 0.5,
    ):
        self.gpu = gpu
        self.audit_log_path = Path(audit_log_path) if audit_log_path else (
            _results_dir() / "audit_log.jsonl"
        )
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Lazy-init agents
        self._vision:     Optional[VisionAgent]     = None
        self._prompt:     Optional[PromptAgent]      = None
        self._risk:       Optional[RiskAgent]        = None
        self._governance: Optional[GovernanceAgent]  = None

        self._models_dir = models_dir
        self._rules_path = rules_path
        self._prompt_threshold = prompt_threshold

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_vision(self) -> VisionAgent:
        if self._vision is None:
            self._vision = VisionAgent(gpu=self.gpu)
        return self._vision

    def _get_prompt(self) -> PromptAgent:
        if self._prompt is None:
            self._prompt = PromptAgent(
                model_dir=self._models_dir,
                threshold=self._prompt_threshold,
            )
        return self._prompt

    def _get_risk(self) -> RiskAgent:
        if self._risk is None:
            self._risk = RiskAgent(models_dir=self._models_dir)
        return self._risk

    def _get_governance(self) -> GovernanceAgent:
        if self._governance is None:
            self._governance = GovernanceAgent(rules_path=self._rules_path)
        return self._governance

    # ── Main process ──────────────────────────────────────────────────────────

    def process(
        self,
        image_path: Optional[str] = None,
        text: Optional[str] = None,
        sample_id: Optional[str] = None,
        severity: str = "medium",
    ) -> FinalDecision:
        """
        Run the full governance pipeline on a document.

        Args:
            image_path: Path to document image (PNG/JPG/PDF).
            text:       Pre-extracted text (if available).
            sample_id:  Identifier for audit log.
            severity:   Metadata severity hint (low/medium/high/critical).

        Returns FinalDecision with decision, scores, reason, and audit_log_id.
        """
        if image_path is None and text is None:
            raise ValueError("Provide at least one of: image_path, text")

        pipeline_start = time.perf_counter()
        audit_id = str(uuid.uuid4())[:8].upper()
        sid = sample_id or audit_id

        # ── Step 1: Vision Agent ──────────────────────────────────────────────
        vision_ms = 0.0
        vision_error = None
        vision_features_dict = {}
        vision_score = 0.0
        hidden_text_detected = False

        if image_path:
            try:
                t0 = time.perf_counter()
                vf = self._get_vision().extract(image_path, sample_id=sid)
                vision_ms = (time.perf_counter() - t0) * 1000
                vision_features_dict = vf.to_dict()
                vision_score = vf.vision_score
                hidden_text_detected = vf.hidden_text_score >= 0.5
                vision_error = vf.error

                # If no text provided, use OCR from vision agent
                if text is None and not vf.ocr_confidence == 0.0:
                    from ocr_adapter import OCREngineFactory
                    ocr = OCREngineFactory.create(gpu=self.gpu)
                    ocr_result = ocr.run(image_path)
                    text = ocr_result.text
            except Exception as exc:
                vision_error = str(exc)
                vision_score = 0.5   # fail-safe: elevated uncertainty
                print(f"[DecisionAgent] VisionAgent failed: {exc}")

        # ── Step 2: Prompt Agent ──────────────────────────────────────────────
        prompt_ms = 0.0
        prompt_error = None
        prompt_score = 0.5   # conservative default
        prompt_confidence = 0.5

        if text:
            try:
                t0 = time.perf_counter()
                pred = self._get_prompt().predict(text or "", sample_id=sid)
                prompt_ms = (time.perf_counter() - t0) * 1000
                prompt_score = pred.malicious_probability
                prompt_confidence = pred.confidence
            except Exception as exc:
                prompt_error = str(exc)
                prompt_score = 0.5   # fail-safe
                print(f"[DecisionAgent] PromptAgent failed: {exc}")

        # ── Step 3: Risk Agent ────────────────────────────────────────────────
        risk_ms = 0.0
        risk_score = 0.5
        risk_level = "MEDIUM"

        try:
            from risk_agent import RiskAgent as RA
            features_dict = RA.build_feature_dict(
                malicious_probability=prompt_score,
                vision_features=vision_features_dict,
                severity=severity,
            )
            t0 = time.perf_counter()
            risk_result = self._get_risk().score(features_dict, sample_id=sid)
            risk_ms = (time.perf_counter() - t0) * 1000
            risk_score = risk_result.risk_score
            risk_level = risk_result.risk_level
        except Exception as exc:
            # Fail-safe: if risk model not trained yet, use heuristic
            risk_score = max(prompt_score, vision_score)
            risk_level = "HIGH" if risk_score >= 0.65 else "MEDIUM"
            print(f"[DecisionAgent] RiskAgent fallback (model not found): {exc}")

        # ── Step 4: Governance Agent ──────────────────────────────────────────
        gov_ms = 0.0
        gov: GovernanceDecision

        try:
            t0 = time.perf_counter()
            gov = self._get_governance().evaluate(
                risk_score=risk_score,
                vision_score=vision_score,
                hidden_text_detected=hidden_text_detected,
                keyword_density=vision_features_dict.get("keyword_density", 0.0),
                footer_density=vision_features_dict.get("footer_text_density", 0.0),
                watermark_score=vision_features_dict.get("watermark_score", 0.0),
                prompt_score=prompt_score,
                severity=severity,
            )
            gov_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            # Fail-safe: unknown governance state → block
            print(f"[DecisionAgent] GovernanceAgent failed: {exc}")
            from governance_agent import GovernanceDecision as GD
            gov = GD(
                decision="BLOCK",
                rule_id="G_FAILSAFE",
                rule_name="Governance Failure Failsafe",
                severity="CRITICAL",
                policy_ref="NIST AI RMF: GOVERN 1.7",
                reason=f"Governance agent failed ({exc}). Blocking as failsafe.",
                sanitization_action=None,
            )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        # ── Step 5: Assemble result ───────────────────────────────────────────
        result = FinalDecision(
            audit_log_id=audit_id,
            sample_id=sid,
            timestamp=datetime.now(timezone.utc).isoformat(),
            decision=gov.decision,
            risk_score=risk_score,
            risk_level=risk_level,
            prompt_score=prompt_score,
            vision_score=vision_score,
            confidence=prompt_confidence,
            governance_rule_triggered=gov.rule_id,
            reason=gov.reason,
            sanitization_action=gov.sanitization_action,
            policy_ref=gov.policy_ref,
            vision_ms=vision_ms,
            prompt_ms=prompt_ms,
            risk_ms=risk_ms,
            governance_ms=gov_ms,
            total_ms=total_ms,
            vision_error=vision_error,
            prompt_error=prompt_error,
            agent_version=AGENT_VERSION,
        )

        # ── Step 6: Audit log ─────────────────────────────────────────────────
        self._write_audit_log(result)
        return result

    def _write_audit_log(self, result: FinalDecision) -> None:
        """Append JSON Lines entry to audit log."""
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception as exc:
            print(f"[DecisionAgent] Audit log write failed: {exc}")
