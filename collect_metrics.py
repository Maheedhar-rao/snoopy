#!/usr/bin/env python3
"""
Collect daily model performance metrics and upsert to model_metrics table.

Designed to run as a daily cron job or Jenkins task.

Usage:
    python collect_metrics.py                  # collect yesterday's metrics
    python collect_metrics.py --date 2026-02-12  # specific date
    python collect_metrics.py --check-alerts   # also print alert thresholds
"""

import argparse
import json
import os
import sys
import statistics
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")

# Alert thresholds
ALERT_SAFETY_NET_PCT = 2.0
ALERT_FALLBACK_PCT = 25.0
ALERT_ACCURACY_MIN = 0.93

# Cost tracking — Claude Haiku 4.5 (~700 input + ~150 output tokens per call)
# $1/M input + $5/M output ≈ $0.00145/call. Using $0.003 as conservative default.
CLAUDE_COST_PER_EMAIL = float(os.environ.get("CLAUDE_COST_PER_EMAIL", "0.003"))


def _get_model_version() -> str:
    """Read current active model version from meta.json."""
    meta_path = PROJECT_DIR / "models" / "email_classifier" / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f).get("version", "v1")
    return "v1"


def collect(target_date: str = None, check_alerts: bool = False):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE in .env")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    if not target_date:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        target_date = yesterday.strftime("%Y-%m-%d")

    date_start = f"{target_date}T00:00:00Z"
    date_end = f"{target_date}T23:59:59Z"
    model_version = _get_model_version()

    print(f"Collecting metrics for {target_date} (model {model_version})")

    # Fetch day's email_responses
    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        result = (
            sb.table("email_responses")
            .select("id, response_type, confidence, classification_method, lender_name, received_at")
            .gte("received_at", date_start)
            .lte("received_at", date_end)
            .order("id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data or []
        if not batch:
            break
        all_rows.extend(batch)
        offset += page_size

    if not all_rows:
        print(f"  No email responses for {target_date}")
        return

    total = len(all_rows)
    print(f"  Found {total} email responses")

    # Volume breakdown
    methods = Counter(r.get("classification_method", "unknown") for r in all_rows)
    bert_used = methods.get("distilbert", 0)
    claude_fallback = methods.get("ai", 0) + methods.get("claude", 0)
    regex_fallback = methods.get("regex_fallback", 0) + methods.get("regex", 0)

    # Safety net fires
    safety_net_count = 0
    try:
        sn_result = (
            sb.table("classification_feedback")
            .select("id", count="exact")
            .eq("correction_source", "claude_safety_net")
            .gte("created_at", date_start)
            .lte("created_at", date_end)
            .execute()
        )
        safety_net_count = sn_result.count or 0
    except Exception:
        pass

    # Confidence distribution
    confidences = [r.get("confidence", 0) or 0 for r in all_rows if r.get("confidence") is not None]
    conf_mean = statistics.mean(confidences) if confidences else 0
    conf_median = statistics.median(confidences) if confidences else 0
    conf_p10 = sorted(confidences)[int(len(confidences) * 0.1)] if len(confidences) >= 10 else 0
    conf_above_85 = sum(1 for c in confidences if c >= 0.85)
    conf_above_95 = sum(1 for c in confidences if c >= 0.95)

    # Per-class breakdown
    per_class = {}
    class_counts = Counter(r.get("response_type", "OTHER") for r in all_rows)
    for cls, cnt in class_counts.items():
        cls_confs = [r.get("confidence", 0) or 0 for r in all_rows if r.get("response_type") == cls]
        per_class[cls] = {
            "count": cnt,
            "mean_confidence": round(statistics.mean(cls_confs), 4) if cls_confs else 0,
        }

    # Per-lender breakdown
    per_lender = {}
    lender_counts = Counter(r.get("lender_name", "unknown") for r in all_rows)
    for lender, cnt in lender_counts.most_common(20):
        if not lender or lender == "unknown":
            continue
        per_lender[lender] = {"count": cnt}

    # Verified accuracy
    verified_total = 0
    verified_correct = 0
    try:
        fb_result = (
            sb.table("classification_feedback")
            .select("bert_prediction, corrected_label, correction_source")
            .in_("correction_source", ["human_manual", "human_reclassify"])
            .gte("created_at", date_start)
            .lte("created_at", date_end)
            .execute()
        )
        for fb in (fb_result.data or []):
            verified_total += 1
            if fb.get("bert_prediction") == fb.get("corrected_label"):
                verified_correct += 1
    except Exception:
        pass

    verified_accuracy = round(verified_correct / verified_total, 4) if verified_total > 0 else None

    # Feedback count
    feedback_count = 0
    try:
        fc_result = (
            sb.table("classification_feedback")
            .select("id", count="exact")
            .gte("created_at", date_start)
            .lte("created_at", date_end)
            .execute()
        )
        feedback_count = fc_result.count or 0
    except Exception:
        pass

    # Cost calculations
    claude_calls = claude_fallback + safety_net_count
    claude_cost = round(claude_calls * CLAUDE_COST_PER_EMAIL, 4)
    hypothetical_cost = round(total * CLAUDE_COST_PER_EMAIL, 4)  # if 100% Claude
    bert_savings = round(hypothetical_cost - claude_cost, 4)
    savings_pct = round(bert_savings / hypothetical_cost * 100, 1) if hypothetical_cost > 0 else 0.0

    # Upsert to model_metrics
    row = {
        "model_version": model_version,
        "metric_date": target_date,
        "total_classifications": total,
        "bert_used": bert_used,
        "claude_fallback": claude_fallback,
        "safety_net_fired": safety_net_count,
        "regex_fallback": regex_fallback,
        "conf_mean": round(conf_mean, 4),
        "conf_median": round(conf_median, 4),
        "conf_p10": round(conf_p10, 4),
        "conf_above_85": conf_above_85,
        "conf_above_95": conf_above_95,
        "verified_total": verified_total,
        "verified_correct": verified_correct,
        "verified_accuracy": verified_accuracy,
        "per_class": per_class,
        "per_lender": per_lender,
        "feedback_count": feedback_count,
        "claude_cost": claude_cost,
        "hypothetical_cost": hypothetical_cost,
        "bert_savings": bert_savings,
        "savings_pct": savings_pct,
        "cost_per_email": CLAUDE_COST_PER_EMAIL,
        "tenant_id": "pathway",
    }

    row = {k: v for k, v in row.items() if v is not None}

    try:
        sb.table("model_metrics").upsert(row, on_conflict="model_version,metric_date,tenant_id").execute()
        print(f"  Metrics saved to model_metrics table")
    except Exception as e:
        print(f"  Error saving metrics: {e}")
        print(json.dumps(row, indent=2, default=str))

    # Print summary
    print(f"\n  Summary for {target_date}:")
    print(f"    Total classifications: {total}")
    print(f"    BERT used:            {bert_used} ({bert_used/total*100:.0f}%)")
    print(f"    Claude fallback:      {claude_fallback} ({claude_fallback/total*100:.0f}%)")
    print(f"    Safety net fired:     {safety_net_count}")
    print(f"    Regex fallback:       {regex_fallback}")
    print(f"    Confidence mean:      {conf_mean:.3f}")
    print(f"    Confidence >= 0.85:   {conf_above_85}/{total} ({conf_above_85/total*100:.0f}%)")
    if verified_total > 0:
        print(f"    Verified accuracy:    {verified_correct}/{verified_total} ({verified_accuracy*100:.1f}%)")
    print(f"    Feedback collected:   {feedback_count}")

    # Cost summary
    print(f"\n  Cost breakdown:")
    print(f"    Claude API calls:     {claude_calls} (fallback {claude_fallback} + safety net {safety_net_count})")
    print(f"    Actual Claude cost:   ${claude_cost:.4f}")
    print(f"    If 100% Claude:       ${hypothetical_cost:.4f}")
    print(f"    BERT savings today:   ${bert_savings:.4f} ({savings_pct:.1f}%)")
    print(f"    Cost per email avg:   ${claude_cost/total:.6f}" if total > 0 else "")

    # Alerts
    alerts = []

    if total > 0:
        safety_pct = safety_net_count / total * 100
        if safety_pct > ALERT_SAFETY_NET_PCT:
            alerts.append(f"SAFETY NET rate ({safety_pct:.1f}%) exceeds {ALERT_SAFETY_NET_PCT}%")

        fallback_pct = claude_fallback / total * 100
        if fallback_pct > ALERT_FALLBACK_PCT:
            alerts.append(f"CLAUDE FALLBACK rate ({fallback_pct:.1f}%) exceeds {ALERT_FALLBACK_PCT}%")

    if verified_accuracy is not None and verified_accuracy < ALERT_ACCURACY_MIN:
        alerts.append(f"VERIFIED ACCURACY ({verified_accuracy*100:.1f}%) below {ALERT_ACCURACY_MIN*100:.0f}%")

    if alerts or check_alerts:
        print(f"\n  Alerts:")
        if alerts:
            for a in alerts:
                print(f"    *** {a}")
        else:
            print(f"    All clear - no alerts triggered")

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect daily model metrics")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD), default: yesterday")
    parser.add_argument("--check-alerts", action="store_true", help="Print alert status")
    args = parser.parse_args()

    collect(target_date=args.date, check_alerts=args.check_alerts)
