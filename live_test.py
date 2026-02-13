#!/usr/bin/env python3
"""
Live test: show DistilBERT classification for recent emails with full detail.

Usage:
    python live_test.py              # last 100 emails
    python live_test.py --limit 50   # last 50 emails
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

from supabase import create_client
from predict import get_classifier

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")


def run(limit=100):
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Loading DistilBERT model...")
    bert = get_classifier()

    result = (
        sb.table("email_responses")
        .select("id, subject, body, summary, response_type, classification_method, from_email")
        .in_("response_type", ["APPROVED", "DECLINED", "STIPS_REQUIRED", "OTHER"])
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )

    rows = result.data or []
    print(f"Testing {len(rows)} emails\n")

    header = f"{'#':>3}  {'SUBJECT':<55}  {'CURRENT':>15}  {'BERT':>15}  {'CONF':>6}  {'OK':>3}"
    print(header)
    print("-" * len(header))

    correct = 0
    for i, row in enumerate(rows):
        subject = (row.get("subject") or "(no subject)").strip()
        display_subject = subject[:53] if len(subject) > 53 else subject
        body = (row.get("body") or "").strip()
        summary = (row.get("summary") or "").strip()
        ground_truth = row.get("response_type", "")

        r = bert.predict(subject=subject, body=body, snippet=summary)
        match = r["response_type"] == ground_truth
        if match:
            correct += 1
        mark = "Y" if match else "X"

        print(f"{i+1:>3}  {display_subject:<55}  {ground_truth:>15}  {r['response_type']:>15}  {r['confidence']:>6.3f}  {mark:>3}")

    print(f"\nAccuracy: {correct}/{len(rows)} ({correct/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    run(limit=args.limit)
