#!/usr/bin/env python3
"""
Live monitor: watch today's email responses with DistilBERT classification.

Modes:
  CLI:  python live_monitor.py                # terminal dashboard, refresh 30s
        python live_monitor.py --once         # single run
        python live_monitor.py --interval 10  # custom refresh

  Web:  python live_monitor.py --web          # FastAPI service on $PORT
        GET /monitor          — full monitor JSON
        GET /monitor/summary  — stats + cost only
        GET /health           — service health
"""

import json
import os
import sys
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

from supabase import create_client
from predict import get_classifier

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")

# Cost per Claude API call (same as collect_metrics.py)
CLAUDE_COST_PER_EMAIL = float(os.environ.get("CLAUDE_COST_PER_EMAIL", "0.003"))

# ANSI colors (CLI mode only)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

TYPE_COLORS = {
    "APPROVED": GREEN,
    "DECLINED": RED,
    "STIPS_REQUIRED": YELLOW,
    "OTHER": DIM,
    "RECEIVED": CYAN,
}


def colorize(label):
    c = TYPE_COLORS.get(label, "")
    return f"{c}{label}{RESET}"


HISTORY_DIR = PROJECT_DIR / "data" / "monitor_history"


def history_path(date_str):
    """Return the JSONL history file path for a given UTC date."""
    return HISTORY_DIR / f"{date_str}.jsonl"


def load_history(date_str, model_version):
    """Load cached classifications from JSONL. Skips corrupted lines and stale model versions."""
    path = history_path(date_str)
    cache = {}
    if not path.exists():
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("bert_model_version") != model_version:
                continue
            eid = entry.get("id")
            if eid is not None:
                cache[eid] = entry
    return cache


def save_history_record(fh, row, bert_result):
    """Write one classification to the JSONL history file. Crash-safe via flush+fsync."""
    entry = {
        "id": row.get("id"),
        "deal_id": row.get("deal_id"),
        "subject": row.get("subject"),
        "body": row.get("body"),
        "summary": row.get("summary"),
        "from_email": row.get("from_email"),
        "response_type": row.get("response_type"),
        "confidence": row.get("confidence"),
        "classification_method": row.get("classification_method"),
        "received_at": row.get("received_at"),
        "lender_name": row.get("lender_name"),
        "bert_response_type": bert_result["response_type"],
        "bert_confidence": bert_result["confidence"],
        "bert_reason": bert_result.get("reason", ""),
        "bert_approval_flag": bert_result.get("approval_flag", False),
        "bert_evidence": bert_result.get("evidence", {}),
        "bert_all_scores": bert_result.get("all_scores", {}),
        "bert_model_version": bert_result.get("model_version", "v1"),
        "classified_at": datetime.now(timezone.utc).isoformat(),
    }
    fh.write(json.dumps(entry, default=str) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
    return entry


# ── Core data logic (shared by CLI + web) ──

def get_monitor_data(sb, bert, history_cache, history_fh):
    """Fetch today's emails, classify with BERT, return structured data dict."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    result = (
        sb.table("email_responses")
        .select("id, deal_id, subject, body, summary, from_email, response_type, confidence, classification_method, received_at, lender_name")
        .gte("received_at", f"{today}T00:00:00Z")
        .order("received_at", desc=True)
        .limit(200)
        .execute()
    )

    rows = result.data or []

    if not rows:
        return {
            "date": today,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_emails": 0,
            "emails": [],
            "stats": {},
            "cost": {},
            "safety_net_alerts": [],
            "disagreements": [],
        }

    # Snapshot of already-known IDs before this refresh
    known_ids = set(history_cache.keys())

    classified = []
    for row in rows:
        rid = row.get("id")
        cached = history_cache.get(rid)
        if cached is not None:
            r = {
                "response_type": cached["bert_response_type"],
                "confidence": cached["bert_confidence"],
                "reason": cached.get("bert_reason", ""),
                "approval_flag": cached.get("bert_approval_flag", False),
                "evidence": cached.get("bert_evidence", {}),
                "all_scores": cached.get("bert_all_scores", {}),
                "model_version": cached.get("bert_model_version", "v1"),
                "classification_method": "distilbert",
            }
        else:
            subject = (row.get("subject") or "(no subject)").strip()
            body = (row.get("body") or "").strip()
            summary = (row.get("summary") or "").strip()
            r = bert.predict(subject=subject, body=body, snippet=summary)
            entry = save_history_record(history_fh, row, r)
            history_cache[rid] = entry
        classified.append((row, r))

    # Build stats
    new_count = 0
    matches = 0
    high_conf = 0
    disagreements = []
    flagged_approvals = []
    claude_calls = 0
    bert_calls = 0

    emails = []
    for row, r in classified:
        rid = row.get("id")
        is_new = rid not in known_ids
        if is_new:
            new_count += 1

        current_type = row.get("response_type") or "---"
        bert_type = r["response_type"]
        bert_conf = r["confidence"]
        approval_flag = r.get("approval_flag", False)

        match = current_type == bert_type
        if match:
            matches += 1
        else:
            disagreements.append({
                "id": rid,
                "subject": (row.get("subject") or "")[:80],
                "lender": row.get("lender_name") or row.get("from_email") or "unknown",
                "current_type": current_type,
                "current_method": row.get("classification_method") or "unknown",
                "bert_type": bert_type,
                "bert_confidence": bert_conf,
                "bert_reason": r.get("reason", "")[:80],
                "approval_flag": approval_flag,
            })

        if bert_conf >= 0.85:
            high_conf += 1

        if approval_flag:
            flagged_approvals.append({
                "id": rid,
                "subject": (row.get("subject") or "")[:80],
                "lender": row.get("lender_name") or row.get("from_email") or "unknown",
                "bert_type": bert_type,
                "bert_confidence": bert_conf,
                "approval_signals": r.get("evidence", {}).get("APPROVED", {}).get("signals", [])[:4],
            })

        method = (row.get("classification_method") or "").lower()
        if method in ("ai", "claude", "claude_api"):
            claude_calls += 1
        else:
            bert_calls += 1

        received = row.get("received_at") or ""
        time_str = ""
        if received:
            try:
                dt = datetime.fromisoformat(received.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M")
            except Exception:
                time_str = received[:5]

        emails.append({
            "id": rid,
            "time": time_str,
            "lender": row.get("lender_name") or row.get("from_email") or "---",
            "subject": (row.get("subject") or "(no subject)").strip()[:80],
            "current_type": current_type,
            "current_method": row.get("classification_method") or "unknown",
            "bert_type": bert_type,
            "bert_confidence": round(bert_conf, 4),
            "match": match,
            "approval_flag": approval_flag,
            "reason": r.get("reason", "")[:80],
            "is_new": is_new,
        })

    total = len(classified)

    # Cost
    actual_cost = round(claude_calls * CLAUDE_COST_PER_EMAIL, 4)
    hypothetical_cost = round(total * CLAUDE_COST_PER_EMAIL, 4)
    savings = round(hypothetical_cost - actual_cost, 4)
    savings_pct = round((savings / hypothetical_cost * 100), 1) if hypothetical_cost > 0 else 0

    return {
        "date": today,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_emails": total,
        "emails": emails,
        "stats": {
            "matches": matches,
            "agreement_pct": round(matches / total * 100, 1) if total > 0 else 0,
            "high_confidence": high_conf,
            "high_confidence_pct": round(high_conf / total * 100, 1) if total > 0 else 0,
            "new_since_refresh": new_count,
            "bert_calls": bert_calls,
            "claude_calls": claude_calls,
        },
        "cost": {
            "actual_claude_cost": actual_cost,
            "hypothetical_all_claude": hypothetical_cost,
            "bert_savings": savings,
            "savings_pct": savings_pct,
            "cost_per_email": CLAUDE_COST_PER_EMAIL,
        },
        "safety_net_alerts": flagged_approvals,
        "disagreements": disagreements[:20],
        "model_version": bert.model_version,
    }


# ── CLI display ──

def display_cli(data):
    """Print ANSI-colored terminal output from monitor data."""
    print("\033[2J\033[H", end="")

    now = datetime.now().strftime("%H:%M:%S")
    total = data["total_emails"]
    print(f"{BOLD}Live Email Classification Monitor{RESET}  |  {now}  |  {total} emails today")
    print(f"{DIM}DistilBERT threshold: 0.85 -- above = use BERT (free), below = Claude API fallback{RESET}")
    print()

    if total == 0:
        print("  No email responses received today yet.")
        return

    hdr = (
        f"  {'TIME':>5}  "
        f"{'LENDER':<16}  "
        f"{'SUBJECT':<35}  "
        f"{'CURRENT':>15}  "
        f"{'BERT':>15}  "
        f"{'CONF':>6}  "
        f"{'':>3}  "
        f"{'REASON':<40}"
    )
    print(hdr)
    print("-" * len(hdr))

    for e in data["emails"]:
        lender = e["lender"][:14]
        subj = e["subject"][:33]
        match_icon = f"{GREEN}Y{RESET}" if e["match"] else f"{RED}X{RESET}"
        flag_icon = f" {RED}{BOLD}!{RESET}" if e["approval_flag"] else ""
        prefix = f"{CYAN}*{RESET}" if e["is_new"] else " "

        print(
            f"{prefix} {e['time']:>5}  "
            f"{lender:<16}  "
            f"{subj:<35}  "
            f"{colorize(e['current_type']):>24}  "
            f"{colorize(e['bert_type']):>24}  "
            f"{e['bert_confidence']:>6.3f}  "
            f" {match_icon}{flag_icon}  "
            f"{DIM}{e['reason'][:40]}{RESET}"
        )

    stats = data["stats"]
    cost = data["cost"]

    print()
    print(
        f"  {BOLD}Agreement:{RESET} {stats['matches']}/{total} ({stats['agreement_pct']:.0f}%)   "
        f"{BOLD}High confidence (>=0.85):{RESET} {stats['high_confidence']}/{total} ({stats['high_confidence_pct']:.0f}%)"
    )

    # Cost summary
    savings_pct = cost["savings_pct"]
    savings_color = GREEN if savings_pct >= 50 else (YELLOW if savings_pct >= 20 else RED)
    print(
        f"  {BOLD}Cost today:{RESET} "
        f"${cost['actual_claude_cost']:.4f} actual  |  "
        f"${cost['hypothetical_all_claude']:.4f} if all Claude  |  "
        f"{savings_color}${cost['bert_savings']:.4f} saved ({savings_pct:.0f}%){RESET}  "
        f"{DIM}[BERT:{stats['bert_calls']} Claude:{stats['claude_calls']}]{RESET}"
    )

    if stats["new_since_refresh"] > 0:
        print(f"  {CYAN}* {stats['new_since_refresh']} new since last refresh{RESET}")

    if data["safety_net_alerts"]:
        print(f"\n  {BOLD}{RED}! APPROVAL SAFETY NET -- {len(data['safety_net_alerts'])} emails would go to Claude API:{RESET}")
        for alert in data["safety_net_alerts"]:
            print(
                f"    {BOLD}{alert['lender'][:20]}{RESET}: BERT={colorize(alert['bert_type'])} "
                f"(conf {alert['bert_confidence']:.2f})  "
                f"approval evidence: {GREEN}{', '.join(alert['approval_signals'])}{RESET}"
            )
            print(f"      {DIM}{alert['subject'][:60]}{RESET}")

    if data["disagreements"]:
        print(f"\n  {BOLD}{YELLOW}Disagreements ({len(data['disagreements'])}):{RESET}")
        for d in data["disagreements"][:8]:
            flag = f" {RED}!{RESET}" if d.get("approval_flag") else ""
            print(
                f"    {colorize(d['current_type']):>24} -> BERT: {colorize(d['bert_type']):>24} "
                f"(conf {d['bert_confidence']:.2f}, was {d['current_method']}){flag}"
            )
            print(f"      {DIM}reason: {d['bert_reason'][:50]}  |  {d['subject'][:55]}{RESET}")


# ── CLI main loop ──

def run_cli(once=False, interval=30):
    """Run the monitor in terminal mode."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE in .env")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Loading DistilBERT model...")
    bert = get_classifier()
    print("Model loaded. Starting monitor...\n")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history_cache = load_history(current_date, bert.model_version)
    history_fh = open(history_path(current_date), "a", encoding="utf-8")

    if history_cache:
        print(f"  Resumed {len(history_cache)} cached classifications from history.\n")

    if once:
        data = get_monitor_data(sb, bert, history_cache, history_fh)
        display_cli(data)
        history_fh.close()
        return

    try:
        while True:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != current_date:
                history_fh.close()
                current_date = today
                history_cache = load_history(current_date, bert.model_version)
                history_fh = open(history_path(current_date), "a", encoding="utf-8")

            data = get_monitor_data(sb, bert, history_cache, history_fh)
            display_cli(data)
            print(f"\n  {DIM}Refreshing in {interval}s... (Ctrl+C to stop){RESET}")
            time.sleep(interval)
    except KeyboardInterrupt:
        history_fh.close()
        print(f"\n{DIM}Monitor stopped.{RESET}")


# ── Web mode (FastAPI service) ──

def create_web_app():
    """Create a FastAPI app for the monitor service."""
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("monitor-service")

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE in .env")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.info("Loading DistilBERT model for monitor service...")
    bert = get_classifier()
    logger.info(f"Model loaded: {bert.model_version} on {bert.device}")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # Shared state with lock for thread safety
    _state = {
        "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "history_cache": {},
        "history_fh": None,
    }
    _state_lock = threading.Lock()

    def _get_history_state():
        """Get or refresh history state, handling date rollover."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with _state_lock:
            if today != _state["current_date"] or _state["history_fh"] is None:
                if _state["history_fh"] is not None:
                    _state["history_fh"].close()
                _state["current_date"] = today
                _state["history_cache"] = load_history(today, bert.model_version)
                _state["history_fh"] = open(history_path(today), "a", encoding="utf-8")
            return _state["history_cache"], _state["history_fh"]

    # Initial load
    _get_history_state()
    cached = _state["history_cache"]
    if cached:
        logger.info(f"Resumed {len(cached)} cached classifications from history.")

    monitor_app = FastAPI(
        title="Email Classification Monitor",
        description="Live monitoring dashboard for DistilBERT email classifier",
        version="1.0.0",
    )

    monitor_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @monitor_app.get("/health")
    @monitor_app.get("/v1/health")
    def health():
        return {
            "status": "healthy",
            "service": "monitor",
            "model_version": bert.model_version,
            "device": str(bert.device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @monitor_app.get("/monitor")
    def monitor(limit: int = Query(200, ge=1, le=500)):
        """Full monitor data — emails, stats, cost, alerts."""
        history_cache, history_fh = _get_history_state()
        data = get_monitor_data(sb, bert, history_cache, history_fh)
        data["emails"] = data["emails"][:limit]
        return data

    @monitor_app.get("/monitor/summary")
    def monitor_summary():
        """Quick summary — stats + cost only, no email list."""
        history_cache, history_fh = _get_history_state()
        data = get_monitor_data(sb, bert, history_cache, history_fh)
        return {
            "date": data["date"],
            "timestamp": data["timestamp"],
            "total_emails": data["total_emails"],
            "stats": data["stats"],
            "cost": data["cost"],
            "safety_net_count": len(data["safety_net_alerts"]),
            "disagreement_count": len(data["disagreements"]),
            "model_version": data["model_version"],
        }

    @monitor_app.get("/monitor/disagreements")
    def monitor_disagreements():
        """Just the disagreements between current labels and BERT."""
        history_cache, history_fh = _get_history_state()
        data = get_monitor_data(sb, bert, history_cache, history_fh)
        return {
            "date": data["date"],
            "total_emails": data["total_emails"],
            "disagreements": data["disagreements"],
            "safety_net_alerts": data["safety_net_alerts"],
        }

    @monitor_app.get("/monitor/cost")
    def monitor_cost():
        """Today's cost breakdown."""
        history_cache, history_fh = _get_history_state()
        data = get_monitor_data(sb, bert, history_cache, history_fh)
        return {
            "date": data["date"],
            "total_emails": data["total_emails"],
            "cost": data["cost"],
            "stats": {
                "bert_calls": data["stats"]["bert_calls"],
                "claude_calls": data["stats"]["claude_calls"],
            },
        }

    return monitor_app


# ── Entrypoint ──

def main():
    parser = argparse.ArgumentParser(description="Live email classification monitor")
    parser.add_argument("--once", action="store_true", help="Single run, no auto-refresh")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--web", action="store_true", help="Run as FastAPI web service")
    parser.add_argument("--port", type=int, default=None, help="Port for web mode (default: $PORT or 8081)")
    args = parser.parse_args()

    if args.web:
        import uvicorn
        monitor_app = create_web_app()
        port = args.port or int(os.environ.get("PORT", "8081"))
        uvicorn.run(monitor_app, host="0.0.0.0", port=port)
    else:
        run_cli(once=args.once, interval=args.interval)


# Web app instance for uvicorn direct import: uvicorn live_monitor:web_app
web_app = None

if os.environ.get("MONITOR_WEB_MODE") == "1":
    web_app = create_web_app()


if __name__ == "__main__":
    main()
