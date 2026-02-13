#!/usr/bin/env python3
"""
A/B Test: DistilBERT vs Regex on live email_responses.

Fetches recent emails from Supabase, runs both classifiers,
compares results side by side. No changes to production -- read-only.

Usage:
    python ab_test.py                  # test last 200 emails
    python ab_test.py --limit 500      # test last 500 emails
"""

import os
import sys
import re
import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

from supabase import create_client
from predict import get_classifier


def _regex_classify(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(approved|clear to fund|ctf|green light)\b", t):
        return "APPROVED"
    if re.search(r"\b(declin(e|ed|ing)|cannot|won't|pass(ed|ing)?|not a (good )?fit|no room)\b", t):
        return "DECLINED"
    if re.search(r"\b(stips|stip|need(ed)?|please provide|missing|docs|documents|more info)\b", t):
        return "STIPS_REQUIRED"
    return "OTHER"


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")
REPORT_PATH = PROJECT_DIR / "data" / "ab_test_results.csv"


def run_ab_test(limit: int = 200):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE in .env")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Loading DistilBERT model...")
    bert = get_classifier()
    print("  Model loaded.\n")

    print(f"Fetching {limit} recent classified emails from Supabase...")

    result = (
        sb.table("email_responses")
        .select("id, subject, body, summary, response_type, confidence, classification_method, from_email")
        .in_("response_type", ["APPROVED", "DECLINED", "STIPS_REQUIRED", "OTHER"])
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )

    rows = result.data or []
    print(f"  Fetched {len(rows)} emails.\n")

    if not rows:
        print("No emails to test.")
        return

    results = []
    bert_correct = 0
    regex_correct = 0
    agreements = 0

    print(f"{'#':>4}  {'GROUND TRUTH':>15}  {'DISTILBERT':>15}  {'CONF':>6}  {'REGEX':>15}  {'MATCH':>5}")
    print("-" * 80)

    for i, row in enumerate(rows):
        subject = (row.get("subject") or "").strip()
        body = (row.get("body") or "").strip()
        summary = (row.get("summary") or "").strip()
        ground_truth = row.get("response_type", "")

        bert_result = bert.predict(subject=subject, body=body, snippet=summary)
        bert_type = bert_result["response_type"]
        bert_conf = bert_result["confidence"]

        text = f"{subject}\n{body or summary}"
        regex_type = _regex_classify(text)

        bert_match = bert_type == ground_truth
        regex_match = regex_type == ground_truth
        bert_regex_agree = bert_type == regex_type

        if bert_match:
            bert_correct += 1
        if regex_match:
            regex_correct += 1
        if bert_regex_agree:
            agreements += 1

        mark = "Y" if bert_match else "X"
        if i < 20 or not bert_match:
            print(f"{i+1:>4}  {ground_truth:>15}  {bert_type:>15}  {bert_conf:>6.3f}  {regex_type:>15}  {mark:>5}")

        results.append({
            "id": row.get("id"),
            "subject": subject[:100],
            "ground_truth": ground_truth,
            "distilbert": bert_type,
            "distilbert_conf": round(bert_conf, 4),
            "regex": regex_type,
            "original_method": row.get("classification_method", ""),
            "bert_correct": bert_match,
            "regex_correct": regex_match,
        })

    total = len(results)

    print("\n" + "=" * 80)
    print("A/B TEST RESULTS")
    print("=" * 80)

    print(f"\n  Total emails tested: {total}")
    print(f"\n  DistilBERT accuracy: {bert_correct}/{total} ({bert_correct/total*100:.1f}%)")
    print(f"  Regex accuracy:      {regex_correct}/{total} ({regex_correct/total*100:.1f}%)")
    print(f"  BERT-Regex agreement:{agreements}/{total} ({agreements/total*100:.1f}%)")

    print(f"\n  Per-class accuracy (DistilBERT):")
    class_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        gt = r["ground_truth"]
        class_stats[gt]["total"] += 1
        if r["bert_correct"]:
            class_stats[gt]["correct"] += 1

    for label in ["APPROVED", "DECLINED", "STIPS_REQUIRED", "OTHER"]:
        s = class_stats[label]
        if s["total"] > 0:
            acc = s["correct"] / s["total"] * 100
            print(f"    {label:20s}: {s['correct']:>4}/{s['total']:<4} ({acc:.1f}%)")

    print(f"\n  DistilBERT confidence vs accuracy:")
    for threshold in [0.5, 0.7, 0.85, 0.9, 0.95]:
        above = [r for r in results if r["distilbert_conf"] >= threshold]
        if above:
            correct = sum(1 for r in above if r["bert_correct"])
            acc = correct / len(above) * 100
            coverage = len(above) / total * 100
            print(f"    conf >= {threshold:.2f}: {acc:.1f}% accurate, {coverage:.1f}% coverage ({len(above)}/{total})")

    errors = [r for r in results if not r["bert_correct"]]
    if errors:
        print(f"\n  DistilBERT errors ({len(errors)}):")
        error_patterns = Counter(f"{r['ground_truth']} -> {r['distilbert']}" for r in errors)
        for pattern, count in error_patterns.most_common(10):
            print(f"    {pattern:40s}: {count}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Full results saved to: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test DistilBERT vs Regex")
    parser.add_argument("--limit", type=int, default=200, help="Number of emails to test")
    args = parser.parse_args()

    run_ab_test(limit=args.limit)
