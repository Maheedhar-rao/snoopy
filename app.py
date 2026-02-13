"""
Email Classification Service — Standalone ML API

Runs DistilBERT email classifier as a Railway service.
Endpoints:
  POST /v1/classify          — classify an email
  GET  /v1/health            — health + model info
  GET  /v1/feedback/logs     — browse feedback entries
  GET  /v1/feedback/shadow   — shadow rollout report
  POST /v1/rollout           — change BERT traffic %
  POST /v1/reload            — hot-swap model
"""

import os
import json
import re
import random
import logging
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from predict import get_classifier, reload_classifier

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ml-service")

# ── Config ──
API_KEYS = set(filter(None, os.environ.get("ML_API_KEYS", "").split(",")))
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")
BERT_TRAFFIC_PCT = int(os.environ.get("BERT_TRAFFIC_PCT", "30"))

# ── Supabase client (lazy singleton) ──
_sb = None


def _get_sb():
    global _sb
    if _sb is not None:
        return _sb
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _sb


# ── Anthropic client (lazy singleton) ──
_anthropic_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ── Pre-load model on startup ──
logger.info("Loading model...")
try:
    _clf = get_classifier()
    logger.info(f"Model loaded: {_clf.model_version} on {_clf.device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    _clf = None

# ── FastAPI app ──
app = FastAPI(
    title="Email Classification API",
    description="DistilBERT-powered MCA lender email classifier",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ──
def require_api_key(authorization: Optional[str] = None, x_api_key: Optional[str] = None):
    """Validate API key from Authorization header or X-API-Key header."""
    if not API_KEYS:
        return True  # No keys configured = open access (dev mode)

    key = None
    if authorization and authorization.startswith("Bearer "):
        key = authorization[7:]
    elif x_api_key:
        key = x_api_key

    if not key or key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


# ── Request/Response models ──
class ClassifyRequest(BaseModel):
    subject: str = ""
    body: str = ""
    snippet: str = ""
    # Optional metadata (for feedback logging)
    email_response_id: Optional[int] = None
    deal_id: Optional[int] = None
    from_email: str = ""
    lender_name: str = ""
    tenant_id: str = "pathway"


class ClassifyResponse(BaseModel):
    response_type: str
    confidence: float
    classification_method: str
    model_version: str = "v1"
    reason: str = ""
    evidence: dict = {}
    approval_flag: bool = False
    all_scores: dict = {}
    offer_details: dict = {}
    sentiment: str = "neutral"
    decline_reason: str = ""
    summary: str = ""
    stips_requested: list = []


class RolloutRequest(BaseModel):
    pct: int = Field(ge=0, le=100)


# ── Quality score mapping ──
_QUALITY_SCORES = {
    "claude_shadow": 0.9,
    "claude_fallback": 0.9,
    "claude_safety_net": 0.9,
    "human_manual": 1.0,
    "human_reclassify": 1.0,
}


def _log_feedback_to_db(*, tenant_id="pathway", **kwargs):
    """Log feedback to Supabase (fail-silent)."""
    sb = _get_sb()
    if not sb:
        return
    source = kwargs.get("correction_source", "unknown")
    quality = _QUALITY_SCORES.get(source, 0.8)
    conf = kwargs.get("correction_confidence")
    if source.startswith("claude") and conf and conf < 0.8:
        quality = 0.7

    row = {**kwargs, "quality_score": quality, "tenant_id": tenant_id}
    row = {k: v for k, v in row.items() if v is not None}

    try:
        sb.table("classification_feedback").insert(row).execute()
    except Exception as e:
        logger.warning(f"Failed to log feedback: {e}")


# ── Endpoints ──

@app.post("/v1/classify", response_model=ClassifyResponse)
def classify_email(
    req: ClassifyRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Classify a lender email response."""
    require_api_key(authorization, x_api_key)

    clf = get_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = clf.predict(subject=req.subject, body=req.body, snippet=req.snippet)
    bert_conf = result.get("confidence", 0)
    approval_flag = result.get("approval_flag", False)

    # Gradual rollout logic
    global BERT_TRAFFIC_PCT
    use_bert = (
        not approval_flag
        and bert_conf >= 0.85
        and random.randint(1, 100) <= BERT_TRAFFIC_PCT
    )

    if use_bert:
        # Extract offer details via regex if APPROVED
        offer_details = {}
        if result["response_type"] == "APPROVED":
            text = f"{req.subject}\n{req.body or req.snippet}".lower()
            m = re.search(r"\$\s*([0-9][0-9,]{3,})", text)
            if m:
                try:
                    offer_details["amount"] = int(m.group(1).replace(",", ""))
                except ValueError:
                    pass
            m = re.search(r"(factor|buy rate)[^\d]*([0-9][.]\d{1,2})", text)
            if m:
                try:
                    offer_details["factor_rate"] = float(m.group(2))
                except ValueError:
                    pass

        return ClassifyResponse(
            response_type=result["response_type"],
            confidence=result["confidence"],
            classification_method="distilbert",
            model_version=result.get("model_version", "v1"),
            reason=result.get("reason", ""),
            evidence=result.get("evidence", {}),
            approval_flag=result.get("approval_flag", False),
            all_scores=result.get("all_scores", {}),
            offer_details=offer_details,
            sentiment=result.get("sentiment", "neutral"),
            decline_reason=result.get("decline_reason", ""),
            summary=result.get("summary", ""),
            stips_requested=result.get("stips_requested", []),
        )

    # Fall to Claude API
    claude_result = _classify_with_claude(req.subject, req.body, req.snippet)
    claude_type = claude_result.get("response_type", "OTHER")

    # Log feedback: BERT vs Claude comparison
    if approval_flag:
        source = "claude_safety_net"
    elif bert_conf >= 0.85:
        source = "claude_shadow"
    else:
        source = "claude_fallback"

    _log_feedback_to_db(
        email_response_id=req.email_response_id,
        deal_id=req.deal_id,
        subject=req.subject[:500],
        body=req.body[:5000],
        snippet=req.snippet[:1000],
        from_email=req.from_email[:200],
        lender_name=req.lender_name[:200],
        bert_prediction=result["response_type"],
        bert_confidence=result["confidence"],
        bert_all_scores=result.get("all_scores"),
        corrected_label=claude_type,
        correction_source=source,
        correction_confidence=claude_result.get("confidence"),
        approval_flag_fired=approval_flag,
        evidence_json=result.get("evidence"),
        claude_raw_response=claude_result,
        tenant_id=req.tenant_id,
    )

    # Derive sentiment from Claude's response or fall back to type-based
    claude_sentiment = claude_result.get("sentiment", "")
    if not claude_sentiment:
        _sentiment_map = {"APPROVED": "positive", "DECLINED": "negative"}
        claude_sentiment = _sentiment_map.get(claude_type, "neutral")

    return ClassifyResponse(
        response_type=claude_type,
        confidence=claude_result.get("confidence", 0.5),
        classification_method="claude_api",
        model_version=result.get("model_version", "v1"),
        reason=claude_result.get("summary", ""),
        evidence=result.get("evidence", {}),
        approval_flag=approval_flag,
        all_scores=result.get("all_scores", {}),
        offer_details=claude_result.get("offer_details", {}),
        sentiment=claude_sentiment,
        decline_reason=claude_result.get("decline_reason", ""),
        summary=claude_result.get("summary", ""),
        stips_requested=claude_result.get("stips_requested", []),
    )


def _classify_with_claude(subject: str, body: str, snippet: str) -> dict:
    """Claude API fallback classifier (uses singleton client)."""
    client = _get_anthropic()
    if not client:
        return {"response_type": "OTHER", "confidence": 0.5, "summary": "No Claude API key"}

    try:
        email_content = f"Subject: {subject}\n\nBody:\n{body or snippet or '(empty)'}"

        prompt = f"""Analyze this lender email response and classify it. This is from a lending/financing context where lenders respond to loan applications.

Email:
---
{email_content}
---

Classify this email. Return a JSON object with:
- response_type: "APPROVED", "DECLINED", "STIPS_REQUIRED", or "OTHER"
- confidence: Float 0.0-1.0
- offer_details: {{amount, factor_rate, term, payment}} (for APPROVED only)
- stips_requested: [list] (for STIPS_REQUIRED only)
- decline_reason: string (for DECLINED only)
- summary: 1-2 sentence summary

Return ONLY valid JSON."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        result = json.loads(text)
        result["confidence"] = float(result.get("confidence", 0.5))
        return result

    except Exception as e:
        logger.error(f"Claude classification failed: {e}")
        return {"response_type": "OTHER", "confidence": 0.3, "summary": f"Claude error: {e}"}


@app.get("/v1/health")
def health():
    """Health check with model info and today's cost snapshot."""
    clf = get_classifier()
    model_loaded = clf is not None

    info = {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "bert_traffic_pct": BERT_TRAFFIC_PCT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if model_loaded:
        info["model_version"] = clf.model_version
        info["device"] = str(clf.device)
        info["labels"] = clf.labels

    # Quick cost snapshot from latest metrics
    sb = _get_sb()
    if sb:
        try:
            latest = (
                sb.table("model_metrics")
                .select("metric_date, bert_savings, savings_pct, claude_cost, total_classifications, bert_used")
                .eq("tenant_id", "pathway")
                .order("metric_date", desc=True)
                .limit(1)
                .execute()
            )
            if latest.data:
                m = latest.data[0]
                info["latest_cost"] = {
                    "date": m.get("metric_date"),
                    "total_emails": m.get("total_classifications", 0),
                    "bert_handled": m.get("bert_used", 0),
                    "claude_cost": m.get("claude_cost", 0),
                    "savings": m.get("bert_savings", 0),
                    "savings_pct": m.get("savings_pct", 0),
                }
        except Exception:
            pass

    return info


@app.post("/v1/rollout")
def set_rollout(
    req: RolloutRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Change BERT traffic percentage at runtime."""
    require_api_key(authorization, x_api_key)
    global BERT_TRAFFIC_PCT
    old = BERT_TRAFFIC_PCT
    BERT_TRAFFIC_PCT = req.pct
    logger.info(f"Rollout changed: {old}% -> {req.pct}%")
    return {"old_pct": old, "new_pct": req.pct}


@app.post("/v1/reload")
def reload_model(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Hot-swap model from disk."""
    require_api_key(authorization, x_api_key)
    try:
        new_version = reload_classifier()
        logger.info(f"Model reloaded: {new_version}")
        return {"success": True, "model_version": new_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/feedback/logs")
def get_feedback_logs(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    source: Optional[str] = Query(None),
    deal_id: Optional[int] = Query(None),
    date: Optional[str] = Query(None),
    mismatch: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Browse classification feedback logs."""
    require_api_key(authorization, x_api_key)
    sb = _get_sb()
    if not sb:
        raise HTTPException(status_code=503, detail="Database not configured")

    query = (
        sb.table("classification_feedback")
        .select("id, email_response_id, deal_id, subject, from_email, lender_name, "
                "bert_prediction, bert_confidence, corrected_label, correction_source, "
                "correction_confidence, approval_flag_fired, quality_score, created_at")
        .order("created_at", desc=True)
    )

    if source:
        query = query.eq("correction_source", source)
    if deal_id:
        query = query.eq("deal_id", deal_id)
    if date:
        query = query.gte("created_at", f"{date}T00:00:00Z").lte("created_at", f"{date}T23:59:59Z")

    query = query.range(offset, offset + limit - 1)
    result = query.execute()
    rows = result.data or []

    if mismatch:
        rows = [r for r in rows if r.get("bert_prediction") != r.get("corrected_label")]

    for r in rows:
        r["bert_agrees"] = r.get("bert_prediction") == r.get("corrected_label")

    stats = _build_feedback_stats(sb)

    return {
        "logs": rows,
        "count": len(rows),
        "offset": offset,
        "limit": limit,
        "stats": stats,
        "bert_traffic_pct": BERT_TRAFFIC_PCT,
    }


@app.get("/v1/feedback/shadow")
def get_shadow_report(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    days: int = Query(7, ge=1, le=90),
):
    """Shadow rollout agreement report."""
    require_api_key(authorization, x_api_key)
    sb = _get_sb()
    if not sb:
        raise HTTPException(status_code=503, detail="Database not configured")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    result = (
        sb.table("classification_feedback")
        .select("bert_prediction, bert_confidence, corrected_label, "
                "correction_confidence, lender_name, created_at")
        .eq("correction_source", "claude_shadow")
        .gte("created_at", cutoff)
        .order("created_at", desc=True)
        .limit(1000)
        .execute()
    )
    rows = result.data or []

    if not rows:
        return {"message": "No shadow data yet", "total": 0, "days": days}

    total = len(rows)
    matches = sum(1 for r in rows if r.get("bert_prediction") == r.get("corrected_label"))

    # By class
    by_class = {}
    for r in rows:
        cls = r.get("bert_prediction", "?")
        if cls not in by_class:
            by_class[cls] = {"total": 0, "correct": 0, "misclassified_as": {}}
        by_class[cls]["total"] += 1
        if cls == r.get("corrected_label"):
            by_class[cls]["correct"] += 1
        else:
            mc = r.get("corrected_label", "?")
            by_class[cls]["misclassified_as"][mc] = by_class[cls]["misclassified_as"].get(mc, 0) + 1

    # By lender
    by_lender = {}
    for r in rows:
        lender = r.get("lender_name") or "unknown"
        if lender not in by_lender:
            by_lender[lender] = {"total": 0, "matches": 0}
        by_lender[lender]["total"] += 1
        if r.get("bert_prediction") == r.get("corrected_label"):
            by_lender[lender]["matches"] += 1

    by_lender_sorted = sorted(
        [{"lender": k, **v, "rate": round(v["matches"] / v["total"], 3)}
         for k, v in by_lender.items()],
        key=lambda x: x["total"] - x["matches"], reverse=True,
    )[:15]

    # Recent mismatches
    mismatches = [
        {"bert": r.get("bert_prediction"), "claude": r.get("corrected_label"),
         "bert_conf": r.get("bert_confidence"), "lender": r.get("lender_name"),
         "at": r.get("created_at")}
        for r in rows if r.get("bert_prediction") != r.get("corrected_label")
    ][:20]

    rate = round(matches / total, 4) if total > 0 else 0

    # Recommendation
    if total < 20:
        rec = "Not enough data yet. Wait for more traffic."
    elif rate >= 0.97:
        rec = "Excellent (97%+). Safe to increase rollout by 20%."
    elif rate >= 0.95:
        rec = "Strong (95%+). Safe to increase rollout by 10%."
    elif rate >= 0.90:
        rec = "Good (90%+). Increase by 5%. Review mismatches."
    else:
        rec = f"Below 90% ({rate:.1%}). Hold rollout. Investigate mismatches."

    return {
        "days": days,
        "total": total,
        "matches": matches,
        "agreement_rate": rate,
        "by_class": by_class,
        "by_lender": by_lender_sorted,
        "recent_mismatches": mismatches,
        "recommendation": rec,
        "bert_traffic_pct": BERT_TRAFFIC_PCT,
    }


@app.get("/v1/costs")
def get_costs(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    days: int = Query(30, ge=1, le=365),
):
    """Cost management dashboard — daily Claude spend, BERT savings, trends."""
    require_api_key(authorization, x_api_key)
    sb = _get_sb()
    if not sb:
        raise HTTPException(status_code=503, detail="Database not configured")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    result = (
        sb.table("model_metrics")
        .select("metric_date, total_classifications, bert_used, claude_fallback, "
                "safety_net_fired, claude_cost, hypothetical_cost, bert_savings, "
                "savings_pct, model_version")
        .gte("metric_date", cutoff)
        .eq("tenant_id", "pathway")
        .order("metric_date", desc=True)
        .limit(days)
        .execute()
    )
    rows = result.data or []

    if not rows:
        return {"message": "No cost data yet", "days": days, "daily": []}

    # Aggregate totals
    total_emails = sum(r.get("total_classifications", 0) for r in rows)
    total_bert = sum(r.get("bert_used", 0) for r in rows)
    total_claude_cost = sum(r.get("claude_cost", 0) or 0 for r in rows)
    total_hypothetical = sum(r.get("hypothetical_cost", 0) or 0 for r in rows)
    total_savings = sum(r.get("bert_savings", 0) or 0 for r in rows)

    overall_savings_pct = round(total_savings / total_hypothetical * 100, 1) if total_hypothetical > 0 else 0
    avg_cost_per_email = round(total_claude_cost / total_emails, 6) if total_emails > 0 else 0
    bert_handle_pct = round(total_bert / total_emails * 100, 1) if total_emails > 0 else 0

    # Trend: compare last 7 days to previous 7 days
    sorted_rows = sorted(rows, key=lambda r: r["metric_date"], reverse=True)
    recent_7 = sorted_rows[:7]
    prev_7 = sorted_rows[7:14]

    recent_savings_pct = 0
    if recent_7:
        r_hyp = sum(r.get("hypothetical_cost", 0) or 0 for r in recent_7)
        r_sav = sum(r.get("bert_savings", 0) or 0 for r in recent_7)
        recent_savings_pct = round(r_sav / r_hyp * 100, 1) if r_hyp > 0 else 0

    prev_savings_pct = 0
    if prev_7:
        p_hyp = sum(r.get("hypothetical_cost", 0) or 0 for r in prev_7)
        p_sav = sum(r.get("bert_savings", 0) or 0 for r in prev_7)
        prev_savings_pct = round(p_sav / p_hyp * 100, 1) if p_hyp > 0 else 0

    trend = "improving" if recent_savings_pct > prev_savings_pct else (
        "stable" if recent_savings_pct == prev_savings_pct else "declining"
    )

    # Daily breakdown
    daily = []
    for r in sorted_rows:
        daily.append({
            "date": r["metric_date"],
            "total": r.get("total_classifications", 0),
            "bert_used": r.get("bert_used", 0),
            "claude_calls": (r.get("claude_fallback", 0) or 0) + (r.get("safety_net_fired", 0) or 0),
            "claude_cost": r.get("claude_cost", 0) or 0,
            "hypothetical_cost": r.get("hypothetical_cost", 0) or 0,
            "savings": r.get("bert_savings", 0) or 0,
            "savings_pct": r.get("savings_pct", 0) or 0,
        })

    # Monthly projection
    if len(recent_7) >= 3:
        avg_daily_savings = total_savings / len(rows)
        avg_daily_cost = total_claude_cost / len(rows)
        projected_monthly_savings = round(avg_daily_savings * 30, 2)
        projected_monthly_cost = round(avg_daily_cost * 30, 2)
    else:
        projected_monthly_savings = None
        projected_monthly_cost = None

    return {
        "period_days": days,
        "summary": {
            "total_emails": total_emails,
            "bert_handled": total_bert,
            "bert_handle_pct": bert_handle_pct,
            "total_claude_cost": round(total_claude_cost, 4),
            "hypothetical_all_claude": round(total_hypothetical, 4),
            "total_savings": round(total_savings, 4),
            "savings_pct": overall_savings_pct,
            "avg_cost_per_email": avg_cost_per_email,
        },
        "trend": {
            "direction": trend,
            "last_7d_savings_pct": recent_savings_pct,
            "prev_7d_savings_pct": prev_savings_pct,
        },
        "projection": {
            "monthly_claude_cost": projected_monthly_cost,
            "monthly_savings": projected_monthly_savings,
        },
        "daily": daily,
        "current_bert_traffic_pct": BERT_TRAFFIC_PCT,
    }


def _build_feedback_stats(sb) -> dict:
    """Last-24h feedback summary."""
    try:
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        recent = (
            sb.table("classification_feedback")
            .select("correction_source, bert_prediction, corrected_label")
            .gte("created_at", yesterday)
            .execute()
        )
        rows = recent.data or []

        by_source = {}
        shadow_match = 0
        shadow_total = 0
        for r in rows:
            src = r.get("correction_source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1
            if src == "claude_shadow":
                shadow_total += 1
                if r.get("bert_prediction") == r.get("corrected_label"):
                    shadow_match += 1

        return {
            "last_24h_total": len(rows),
            "by_source": by_source,
            "shadow_agreement": {
                "total": shadow_total,
                "matches": shadow_match,
                "rate": round(shadow_match / shadow_total, 4) if shadow_total > 0 else None,
            },
        }
    except Exception:
        return {}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
