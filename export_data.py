#!/usr/bin/env python3
"""
Export labeled email responses from Supabase for DistilBERT training.

Usage:
    python export_data.py                    # export all high-quality labels
    python export_data.py --min-confidence 0.8  # only high-confidence labels
    python export_data.py --limit 5000       # export subset for quick testing
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")

VALID_LABELS = {"APPROVED", "DECLINED", "STIPS_REQUIRED", "OTHER"}
TRUSTED_METHODS = None  # None = accept all methods

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "email_responses.csv"


def export(min_confidence: float = 0.0, limit: int = 0):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE in .env")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching email_responses from Supabase...")

    # Supabase paginates at 1000 rows — fetch in batches
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        query = (
            sb.table("email_responses")
            .select("subject, body, summary, response_type, confidence, classification_method, from_email")
            .in_("response_type", list(VALID_LABELS))
            .order("id")
            .range(offset, offset + page_size - 1)
        )

        result = query.execute()
        batch = result.data or []

        if not batch:
            break

        all_rows.extend(batch)
        offset += page_size
        print(f"  Fetched {len(all_rows)} rows so far...")

        if limit and len(all_rows) >= limit:
            all_rows = all_rows[:limit]
            break

    print(f"\nTotal rows fetched: {len(all_rows)}")

    # Filter by trusted classification methods and confidence
    filtered = []
    skipped_method = 0
    skipped_confidence = 0
    skipped_empty = 0

    for row in all_rows:
        method = (row.get("classification_method") or "").lower()
        conf = row.get("confidence") or 0.0
        label = row.get("response_type", "")

        if label not in VALID_LABELS:
            continue

        if TRUSTED_METHODS and method and method not in TRUSTED_METHODS:
            skipped_method += 1
            continue

        if min_confidence > 0 and conf < min_confidence:
            skipped_confidence += 1
            continue

        subject = (row.get("subject") or "").strip()
        body = (row.get("body") or "").strip()
        snippet = (row.get("summary") or "").strip()

        if not subject and not body and not snippet:
            skipped_empty += 1
            continue

        text_body = body if body else snippet
        combined = f"Subject: {subject}\n\n{text_body}" if subject else text_body
        combined = combined[:2000]

        filtered.append({
            "text": combined,
            "label": label,
            "confidence": conf,
            "from_email": row.get("from_email", ""),
        })

    print(f"\nFiltered: {len(filtered)} usable rows")
    print(f"  Skipped (method): {skipped_method}")
    print(f"  Skipped (confidence < {min_confidence}): {skipped_confidence}")
    print(f"  Skipped (empty text): {skipped_empty}")

    dist = Counter(r["label"] for r in filtered)
    print(f"\nClass distribution:")
    for label in sorted(dist.keys()):
        count = dist[label]
        pct = (count / len(filtered)) * 100 if filtered else 0
        print(f"  {label:20s}: {count:6d} ({pct:.1f}%)")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "confidence", "from_email"])
        writer.writeheader()
        writer.writerows(filtered)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export email responses for ML training")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence threshold (default: 0.0 = all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max rows to export (default: 0 = all)")
    args = parser.parse_args()

    export(min_confidence=args.min_confidence, limit=args.limit)
