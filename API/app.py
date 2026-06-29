"""
API/app.py
==========
FastAPI REST API for the Agentic AI Governance Framework.

Endpoints:
  POST /upload      — Upload document (image/text) → governance decision
  GET  /health      — System status
  GET  /rules       — List governance rules
  GET  /audit/{id}  — Fetch audit log entry by ID

For Colab demo:
  !pip install fastapi uvicorn python-multipart pyngrok
  !python API/app.py

For local dev:
  uvicorn API.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Add AGENTS/ to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "AGENTS"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from decision_agent import DecisionAgent, FinalDecision
from governance_agent import GovernanceAgent

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic AI Governance Framework",
    description=(
        "Multimodal Prompt Injection Detection for Insurance LLM Pipelines. "
        "Grounded in OWASP LLM Top 10 2025, NIST AI RMF, and ISO 27001."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global agent instance (singleton) ─────────────────────────────────────────

_decision_agent: Optional[DecisionAgent] = None
_governance_agent: Optional[GovernanceAgent] = None
_startup_time = time.time()

def _get_agent() -> DecisionAgent:
    global _decision_agent
    if _decision_agent is None:
        _decision_agent = DecisionAgent(
            gpu=os.environ.get("USE_GPU", "false").lower() == "true"
        )
    return _decision_agent

def _get_governance() -> GovernanceAgent:
    global _governance_agent
    if _governance_agent is None:
        _governance_agent = GovernanceAgent()
    return _governance_agent


# ── Response schemas ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str
    gpu_enabled: bool


class DecisionResponse(BaseModel):
    decision: str
    risk_score: float
    confidence: float
    prompt_score: float
    vision_score: float
    governance_rule_triggered: Optional[str]
    reason: str
    sanitization_action: Optional[str]
    policy_ref: Optional[str]
    processing_time_ms: float
    audit_log_id: str


class RuleResponse(BaseModel):
    id: str
    name: str
    priority: int
    action: str
    severity: str
    policy_ref: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check. Returns uptime and GPU status."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "version": "1.0.0",
        "gpu_enabled": os.environ.get("USE_GPU", "false").lower() == "true",
    }


@app.get("/rules", response_model=list)
async def list_rules():
    """List all active governance rules with policy references."""
    try:
        gov = _get_governance()
        return gov.list_rules()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload", response_model=DecisionResponse)
async def upload_document(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    severity: str = Form("medium"),
    sample_id: Optional[str] = Form(None),
):
    """
    Upload a document image or provide text for governance analysis.

    - **file**: Image file (PNG/JPG) or PDF
    - **text**: Raw text (if no image, or as supplement)
    - **severity**: Metadata hint (low/medium/high/critical)
    - **sample_id**: Optional identifier for audit log

    Returns a governance decision: ALLOW, BLOCK, or SANITIZE.
    """
    if file is None and (text is None or text.strip() == ""):
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: file (image) or text."
        )

    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "application/pdf",
                     "image/webp", "image/tiff"}
    image_path = None

    if file is not None:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {file.content_type}. "
                       f"Allowed: PNG, JPG, PDF, WEBP, TIFF"
            )
        # Validate file size (max 10 MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size is 10 MB."
            )
        if len(contents) == 0:
            raise HTTPException(status_code=422, detail="File is empty.")

        # Write to temp file
        suffix = Path(file.filename).suffix if file.filename else ".png"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(contents)
            image_path = tmp.name

    try:
        agent = _get_agent()
        result: FinalDecision = agent.process(
            image_path=image_path,
            text=text,
            severity=severity,
            sample_id=sample_id,
        )
        return JSONResponse(content=result.to_response_json())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")
    finally:
        # Cleanup temp file
        if image_path and os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except Exception:
                pass


@app.post("/text", response_model=DecisionResponse)
async def analyze_text(
    text: str = Form(...),
    severity: str = Form("medium"),
    sample_id: Optional[str] = Form(None),
):
    """
    Analyze raw text for prompt injection (no image required).
    Faster than /upload — runs only PromptAgent + RiskAgent + GovernanceAgent.
    """
    if not text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")
    try:
        agent = _get_agent()
        result = agent.process(text=text, severity=severity, sample_id=sample_id)
        return JSONResponse(content=result.to_response_json())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    # Auto ngrok tunnel for Colab
    if os.environ.get("USE_NGROK", "false").lower() == "true":
        try:
            from pyngrok import ngrok
            token = os.environ.get("NGROK_TOKEN", "")
            if token:
                ngrok.set_auth_token(token)
            tunnel = ngrok.connect(port)
            print(f"\n🌐 Public URL: {tunnel.public_url}")
            print(f"📋 API Docs:  {tunnel.public_url}/docs\n")
        except ImportError:
            print("pyngrok not installed. Run: pip install pyngrok")

    print(f"🚀 Starting Governance API on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
