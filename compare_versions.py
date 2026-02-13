#!/usr/bin/env python3
"""
Compare two model versions side-by-side on the same test data.

Usage:
    python compare_versions.py v1 v2              # compare v1 vs v2
    python compare_versions.py v1 v2 --data test.csv  # custom test data
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
ACTIVE_DIR = MODELS_DIR / "email_classifier"
TEST_PATH = PROJECT_DIR / "data" / "test_set.csv"

LABELS = ["APPROVED", "DECLINED", "OTHER", "STIPS_REQUIRED"]


def _resolve_model_dir(version: str) -> Path:
    """Resolve version string to model directory path."""
    if version == "active":
        return ACTIVE_DIR

    versioned = MODELS_DIR / f"email_classifier_{version}"
    if versioned.exists():
        return versioned

    p = Path(version)
    if p.exists():
        return p

    print(f"Model not found: {version}")
    print(f"  Checked: {versioned}")
    sys.exit(1)


def _predict_all(model_dir: Path, df: pd.DataFrame, device) -> pd.DataFrame:
    """Run predictions for all rows, return df with predicted/confidence columns."""
    meta_path = model_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    labels = meta["labels"]
    max_length = meta.get("max_length", 512)
    version = meta.get("version", "unknown")

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    all_preds = []
    all_confs = []
    batch_size = 32

    for i in range(0, len(df), batch_size):
        batch_texts = df["text"].iloc[i:i+batch_size].tolist()
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)

        all_preds.extend([labels[p] for p in preds.cpu().tolist()])
        all_confs.extend([round(p, 4) for p in probs.max(dim=-1).values.cpu().tolist()])

    result = df.copy()
    result["predicted"] = all_preds
    result["confidence"] = all_confs
    result["correct"] = result["label"] == result["predicted"]

    del model
    return result, version


def compare(v1: str, v2: str, data_path: str = None):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dir1 = _resolve_model_dir(v1)
    dir2 = _resolve_model_dir(v2)

    test_path = Path(data_path) if data_path else TEST_PATH
    if not test_path.exists():
        print(f"Test data not found: {test_path}")
        sys.exit(1)

    df = pd.read_csv(test_path)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    print(f"Test data: {len(df)} rows from {test_path.name}\n")

    print(f"Running {v1}...")
    df1, ver1 = _predict_all(dir1, df, device)
    print(f"Running {v2}...")
    df2, ver2 = _predict_all(dir2, df, device)

    print(f"\n{'=' * 70}")
    print(f"COMPARISON: {ver1} vs {ver2}")
    print(f"{'=' * 70}")

    acc1 = df1["correct"].mean()
    acc2 = df2["correct"].mean()
    delta = acc2 - acc1
    arrow = "+" if delta > 0 else ""

    print(f"\n  {'Metric':<20s} {ver1:>15s} {ver2:>15s} {'Delta':>10s}")
    print(f"  {'-' * 60}")
    print(f"  {'Accuracy':<20s} {acc1:>15.4f} {acc2:>15.4f} {arrow}{delta:>9.4f}")

    report1 = classification_report(df1["label"], df1["predicted"], labels=LABELS, output_dict=True, zero_division=0)
    report2 = classification_report(df2["label"], df2["predicted"], labels=LABELS, output_dict=True, zero_division=0)

    print(f"\n  Per-class F1:")
    print(f"  {'Class':<20s} {ver1:>10s} {ver2:>10s} {'Delta':>10s}")
    print(f"  {'-' * 50}")
    for l in LABELS:
        f1_1 = report1.get(l, {}).get("f1-score", 0)
        f1_2 = report2.get(l, {}).get("f1-score", 0)
        d = f1_2 - f1_1
        a = "+" if d > 0 else ""
        indicator = " ***" if abs(d) >= 0.02 else ""
        print(f"  {l:<20s} {f1_1:>10.4f} {f1_2:>10.4f} {a}{d:>9.4f}{indicator}")

    macro_f1_1 = report1.get("macro avg", {}).get("f1-score", 0)
    macro_f1_2 = report2.get("macro avg", {}).get("f1-score", 0)
    d = macro_f1_2 - macro_f1_1
    a = "+" if d > 0 else ""
    print(f"  {'Macro F1':<20s} {macro_f1_1:>10.4f} {macro_f1_2:>10.4f} {a}{d:>9.4f}")

    print(f"\n  Confidence stats:")
    print(f"  {'Metric':<20s} {ver1:>10s} {ver2:>10s}")
    print(f"  {'-' * 40}")
    print(f"  {'Mean':<20s} {df1['confidence'].mean():>10.4f} {df2['confidence'].mean():>10.4f}")
    print(f"  {'Median':<20s} {df1['confidence'].median():>10.4f} {df2['confidence'].median():>10.4f}")

    above_85_1 = (df1["confidence"] >= 0.85).mean()
    above_85_2 = (df2["confidence"] >= 0.85).mean()
    print(f"  {'>=0.85 coverage':<20s} {above_85_1:>10.1%} {above_85_2:>10.1%}")

    regressions = df1[df1["correct"] & ~df2["correct"]]
    improvements = df1[~df1["correct"] & df2["correct"]]

    print(f"\n  Regressions (correct in {ver1}, wrong in {ver2}): {len(regressions)}")
    if len(regressions) > 0:
        for _, row in regressions.head(5).iterrows():
            text = row["text"][:80].replace("\n", " ")
            pred2 = df2.loc[row.name, "predicted"]
            print(f"    True={row['label']:15s} {ver2}={pred2:15s} | {text}")

    print(f"\n  Improvements (wrong in {ver1}, correct in {ver2}): {len(improvements)}")
    if len(improvements) > 0:
        for _, row in improvements.head(5).iterrows():
            text = row["text"][:80].replace("\n", " ")
            pred1 = df1.loc[row.name, "predicted"]
            print(f"    True={row['label']:15s} {ver1}={pred1:15s} | {text}")

    print(f"\n{'=' * 70}")
    if acc2 > acc1 and macro_f1_2 >= macro_f1_1:
        print(f"  VERDICT: {ver2} is BETTER (accuracy +{delta:.4f})")
    elif acc2 < acc1:
        print(f"  VERDICT: {ver2} is WORSE (accuracy {delta:.4f})")
    else:
        print(f"  VERDICT: Models are roughly EQUIVALENT")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two model versions")
    parser.add_argument("v1", help="First version (e.g. 'v1', 'active')")
    parser.add_argument("v2", help="Second version (e.g. 'v2')")
    parser.add_argument("--data", help="Path to test CSV")
    args = parser.parse_args()

    compare(args.v1, args.v2, args.data)
