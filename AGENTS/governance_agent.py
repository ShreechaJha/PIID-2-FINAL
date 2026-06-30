"""
AGENTS/governance_agent.py
===========================
Deterministic rule engine — the core research novelty of this project.

Design philosophy (paper Section 4.5 / Section 5):
  Unlike ML models, governance rules are:
  - Auditable: every decision traces to a named rule with policy citation
  - Explainable: a regulator can read the rule and understand the decision
  - Prioritised: BLOCK rules cannot be downgraded by later rules
  - Policy-grounded: all rules cite OWASP LLM Top 10 2025 or NIST AI RMF

Three-tier decision system:
  BLOCK    — high-risk or always-dangerous attack families (hard policy)
  SANITIZE — medium-risk; document may be partially legitimate
  ALLOW    — below all thresholds; passes to the insurance LLM

Governance in regulated industries (insurance) requires decisions be:
  "explainable, auditable, and derived from explicit organizational policy"
  — this is the regulatory argument for rule-based governance over pure ML.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_COLAB_BASE = "/content/drive/MyDrive/PAS"
_LOCAL_BASE = str(Path(__file__).resolve().parents[1])

def _rules_path() -> Path:
    if os.path.exists(_COLAB_BASE):
        return Path(_COLAB_BASE) / "RULES" / "governance_rules.json"
    return Path(_LOCAL_BASE) / "RULES" / "governance_rules.json"


# ── GovernanceDecision ────────────────────────────────────────────────────────

@dataclass
class GovernanceDecision:
    """Output of GovernanceAgent for one document."""
    decision: str               # ALLOW | BLOCK | SANITIZE
    rule_id: Optional[str]      # e.g. "G2" or None if ALLOW (no rule triggered)
    rule_name: Optional[str]
    severity: str               # LOW | MEDIUM | HIGH | CRITICAL
    policy_ref: Optional[str]   # e.g. "OWASP LLM01:2025"
    reason: str                 # Human-readable explanation
    sanitization_action: Optional[str]  # what to strip if SANITIZE

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_blocked(self) -> bool:
        return self.decision == "BLOCK"

    @property
    def is_sanitized(self) -> bool:
        return self.decision == "SANITIZE"


# ── GovernanceAgent ───────────────────────────────────────────────────────────

class GovernanceAgent:
    """
    Evaluates governance rules against extracted signals and produces
    an auditable Allow/Block/Sanitize decision.

    Rules are loaded from RULES/governance_rules.json.
    Rules are evaluated in priority order (lower number = higher priority).
    The first rule that fires determines the decision — no downgrading.

    Usage:
        agent = GovernanceAgent()
        decision = agent.evaluate(
            risk_score=0.91,
            vision_score=0.72,
            hidden_text_detected=True,
            keyword_density=0.45,
            prompt_score=0.88,
            attack_family_predicted="Instruction Override",
            severity="high",
        )
        print(decision.decision, decision.rule_id, decision.reason)
    """

    DEFAULT_ALLOW = GovernanceDecision(
        decision="ALLOW",
        rule_id=None,
        rule_name="Default Allow",
        severity="LOW",
        policy_ref=None,
        reason="No governance rules triggered. Risk score and visual signals "
               "are within acceptable thresholds. Document cleared for processing.",
        sanitization_action=None,
    )

    def __init__(self, rules_path: Optional[str] = None):
        path = Path(rules_path) if rules_path else _rules_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Governance rules not found: {path}\n"
                "Ensure RULES/governance_rules.json exists."
            )
        with open(path) as f:
            raw = json.load(f)

        self._rules: List[Dict[str, Any]] = sorted(
            raw["rules"], key=lambda r: r["priority"]
        )
        self._metadata = raw.get("metadata", {})
        print(f"[GovernanceAgent] Loaded {len(self._rules)} rules "
              f"(version {self._metadata.get('version', 'unknown')})")

    # ── Rule evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        risk_score: float,
        vision_score: float = 0.0,
        hidden_text_detected: bool = False,
        keyword_density: float = 0.0,
        footer_density: float = 0.0,
        watermark_score: float = 0.0,
        prompt_score: float = 0.0,
        attack_family_predicted: str = "unknown",
        severity: str = "medium",
    ) -> GovernanceDecision:
        """
        Evaluate all rules against the provided signals.
        Returns the highest-priority rule that fires, or DEFAULT_ALLOW.

        Parameters mirror the outputs of PromptAgent, VisionAgent, RiskAgent.
        """
        ctx = {
            "risk_score": risk_score,
            "vision_score": vision_score,
            "hidden_text_detected": hidden_text_detected,
            "keyword_density": keyword_density,
            "footer_density": footer_density,
            "watermark_score": watermark_score,
            "prompt_score": prompt_score,
            "attack_family": attack_family_predicted.lower().strip(),
            "severity": severity.lower().strip(),
        }

        for rule in self._rules:
            if self._matches(rule["condition"], ctx):
                reason = rule["reason_template"].format(**ctx)
                sanitize_action = rule.get("sanitization_action")
                return GovernanceDecision(
                    decision=rule["action"],
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    severity=rule["severity"],
                    policy_ref=rule.get("policy_ref"),
                    reason=reason,
                    sanitization_action=sanitize_action,
                )

        return self.DEFAULT_ALLOW

    @staticmethod
    def _matches(condition: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        """
        Evaluates a condition dict against the context.
        Supported operators: gt, gte, lt, lte, eq, in, bool_true
        All sub-conditions must match (implicit AND).
        """
        for field, constraint in condition.items():
            value = ctx.get(field)
            if value is None:
                return False

            if isinstance(constraint, dict):
                op, threshold = list(constraint.items())[0]
                if op == "gt"  and not (value >  threshold): return False
                if op == "gte" and not (value >= threshold): return False
                if op == "lt"  and not (value <  threshold): return False
                if op == "lte" and not (value <= threshold): return False
                if op == "eq"  and not (value == threshold): return False
                if op == "in"  and value not in threshold:   return False
            elif isinstance(constraint, bool):
                if bool(value) != constraint:
                    return False
            elif isinstance(constraint, (int, float)):
                if value != constraint:
                    return False

        return True

    # ── Rule listing (for API /rules endpoint) ────────────────────────────────

    def list_rules(self) -> List[Dict]:
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "priority": r["priority"],
                "action": r["action"],
                "severity": r["severity"],
                "policy_ref": r.get("policy_ref", ""),
            }
            for r in self._rules
        ]
