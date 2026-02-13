#!/usr/bin/env python3
"""
Retrain DistilBERT email classifier with feedback data.

Merges original training data with classification_feedback corrections,
trains a new versioned model, and optionally promotes it to active.

Usage:
    python retrain.py                  # retrain, save new version
    python retrain.py --promote        # auto-promote if metrics improve
    python retrain.py --dry-run        # show data stats only, no training
    python retrain.py --epochs 5       # override training epochs
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
ACTIVE_MODEL_DIR = MODELS_DIR / "email_classifier"

LABELS = ["APPROVED", "DECLINED", "OTHER", "STIPS_REQUIRED"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MAX_LENGTH = 512
MODEL_NAME = "distilbert-base-uncased"


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, weights=None, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.weights = weights
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item


def _text_hash(text: str) -> str:
    """Hash for deduplication."""
    return hashlib.md5(text[:500].encode()).hexdigest()


def _detect_next_version() -> tuple:
    """Scan models directory for existing versions, return (next_version_str, next_version_dir)."""
    existing = []
    if MODELS_DIR.exists():
        for d in MODELS_DIR.iterdir():
            if d.is_dir() and d.name.startswith("email_classifier_v"):
                try:
                    v = int(d.name.split("_v")[-1])
                    existing.append(v)
                except ValueError:
                    pass

    # Also check if the active model has a version in meta.json
    active_meta = ACTIVE_MODEL_DIR / "meta.json"
    if active_meta.exists():
        with open(active_meta) as f:
            meta = json.load(f)
        v_str = meta.get("version", "v1")
        try:
            v_num = int(v_str.lstrip("v"))
            existing.append(v_num)
        except ValueError:
            existing.append(1)

    next_v = max(existing, default=1) + 1
    version_str = f"v{next_v}"
    version_dir = MODELS_DIR / f"email_classifier_{version_str}"
    return version_str, version_dir


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_DIR), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _export_original_data() -> pd.DataFrame:
    """Export original training data from email_responses (same logic as export_data.py)."""
    original_csv = DATA_DIR / "email_responses.csv"
    if original_csv.exists():
        print(f"  Using cached original data: {original_csv}")
        df = pd.read_csv(original_csv)
        df = df[df["label"].isin(LABELS)].reset_index(drop=True)
        return df

    print("  Fetching original data from Supabase...")
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")
    sb = create_client(url, key)

    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        result = (
            sb.table("email_responses")
            .select("subject, body, summary, response_type, confidence, from_email")
            .in_("response_type", LABELS)
            .order("id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data or []
        if not batch:
            break
        all_rows.extend(batch)
        offset += page_size

    rows = []
    for r in all_rows:
        subject = (r.get("subject") or "").strip()
        body = (r.get("body") or "").strip()
        snippet = (r.get("summary") or "").strip()
        label = r.get("response_type", "")
        if label not in LABELS:
            continue
        text_body = body if body else snippet
        if not subject and not text_body:
            continue
        combined = f"Subject: {subject}\n\n{text_body}" if subject else text_body
        rows.append({"text": combined[:2000], "label": label})

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(original_csv, index=False)
    return df


def _export_feedback_data() -> pd.DataFrame:
    """Export untrained feedback from classification_feedback table."""
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("  No Supabase credentials -- skipping feedback export")
        return pd.DataFrame(columns=["text", "label", "quality_score", "source"])

    sb = create_client(url, key)

    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        result = (
            sb.table("classification_feedback")
            .select("id, subject, body, snippet, corrected_label, correction_source, "
                    "correction_confidence, quality_score, used_in_training")
            .eq("used_in_training", False)
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
        return pd.DataFrame(columns=["text", "label", "quality_score", "source", "feedback_id"])

    rows = []
    for r in all_rows:
        subject = (r.get("subject") or "").strip()
        body = (r.get("body") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        label = r.get("corrected_label", "")
        if label not in LABELS:
            continue
        text_body = body if body else snippet
        if not subject and not text_body:
            continue
        combined = f"Subject: {subject}\n\n{text_body}" if subject else text_body
        rows.append({
            "text": combined[:2000],
            "label": label,
            "quality_score": r.get("quality_score", 0.8),
            "source": r.get("correction_source", "unknown"),
            "feedback_id": r.get("id"),
        })

    return pd.DataFrame(rows)


def _mark_feedback_trained(feedback_ids: list, version: str):
    """Mark feedback rows as consumed by this training version."""
    if not feedback_ids:
        return
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return
    sb = create_client(url, key)

    for i in range(0, len(feedback_ids), 100):
        chunk = feedback_ids[i:i+100]
        try:
            sb.table("classification_feedback").update({
                "used_in_training": True,
                "training_version": version,
            }).in_("id", chunk).execute()
        except Exception as e:
            print(f"  Warning: could not mark feedback ids {chunk[:3]}... as trained: {e}")


def _evaluate_model(model, tokenizer, df, device, labels) -> dict:
    """Evaluate model on a dataframe with 'text' and 'label' columns."""
    model.eval()
    all_preds = []
    batch_size = 32

    for i in range(0, len(df), batch_size):
        batch_texts = df["text"].iloc[i:i+batch_size].tolist()
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())

    pred_labels = [labels[p] for p in all_preds]
    true_labels = df["label"].tolist()

    report = classification_report(true_labels, pred_labels, labels=labels, output_dict=True, zero_division=0)
    accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)

    return {
        "accuracy": round(accuracy, 4),
        "per_class": {l: {
            "precision": round(report[l]["precision"], 4),
            "recall": round(report[l]["recall"], 4),
            "f1": round(report[l]["f1-score"], 4),
            "support": report[l]["support"],
        } for l in labels if l in report},
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 4),
    }


SAFETY_NET_MAX_PCT = 2.0  # Promote only if safety net fires on < 2% of test set


def _measure_safety_net_rate(model, tokenizer, df, device, labels) -> dict:
    """
    Run the full predict pipeline (BERT + evidence scanner) on the test set
    and measure how often the approval safety net fires.

    Returns:
        {"total": int, "flagged": int, "rate": float}
    """
    from predict import EmailClassifierLocal

    model.eval()
    flagged = 0
    total = len(df)

    for i in range(total):
        text = df["text"].iloc[i]
        # Run BERT prediction
        enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        label = labels[pred_idx]

        # Run evidence scanner (same as predict.py)
        evidence = EmailClassifierLocal._extract_evidence(text)
        approval_score = evidence["APPROVED"]["score"]
        if label != "APPROVED" and approval_score >= 4:
            flagged += 1

    rate = (flagged / total * 100) if total > 0 else 0.0
    return {"total": total, "flagged": flagged, "rate": round(rate, 2)}


def retrain(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    promote: bool = False,
    dry_run: bool = False,
):
    print("=" * 70)
    print("DistilBERT Email Classifier -- Retrain with Feedback")
    print("=" * 70)

    # Step 1: Detect version
    version_str, version_dir = _detect_next_version()
    print(f"\nNew version: {version_str}")
    print(f"Output dir:  {version_dir}")

    # Step 2: Export data
    print("\n-- Data Export --")
    original_df = _export_original_data()
    print(f"  Original data: {len(original_df)} rows")

    feedback_df = _export_feedback_data()
    print(f"  Feedback data: {len(feedback_df)} rows")

    if len(feedback_df) == 0:
        print("\n  No new feedback to train on. Nothing to do.")
        return

    # Step 3: Merge + deduplicate
    print("\n-- Merge & Deduplicate --")

    original_df["hash"] = original_df["text"].apply(_text_hash)
    feedback_df["hash"] = feedback_df["text"].apply(_text_hash)

    original_hashes = set(original_df["hash"])
    feedback_new = feedback_df[~feedback_df["hash"].isin(original_hashes)]
    feedback_dupes = len(feedback_df) - len(feedback_new)
    print(f"  Feedback after dedup: {len(feedback_new)} (removed {feedback_dupes} duplicates)")

    original_df["weight"] = 1.0
    if "quality_score" in feedback_new.columns:
        feedback_new = feedback_new.copy()
        feedback_new["weight"] = feedback_new["quality_score"].apply(
            lambda q: 2.0 if q >= 1.0 else 1.0
        )
    else:
        feedback_new = feedback_new.copy()
        feedback_new["weight"] = 1.0

    combined_df = pd.concat([
        original_df[["text", "label", "weight", "hash"]],
        feedback_new[["text", "label", "weight", "hash"]],
    ], ignore_index=True)

    combined_df = combined_df.drop_duplicates(subset="hash", keep="last")
    combined_df = combined_df[combined_df["label"].isin(LABELS)].reset_index(drop=True)

    print(f"  Combined dataset: {len(combined_df)} rows")

    dist = Counter(combined_df["label"])
    print(f"  Distribution:")
    for l in LABELS:
        print(f"    {l:20s}: {dist.get(l, 0):6d}")

    if "source" in feedback_new.columns:
        src_dist = Counter(feedback_new["source"])
        print(f"  Feedback sources:")
        for src, cnt in src_dist.most_common():
            print(f"    {src:25s}: {cnt}")

    if dry_run:
        print("\n  --dry-run: stopping before training.")
        return

    # Step 4: Split data
    print("\n-- Train/Val/Test Split --")
    combined_df["label_id"] = combined_df["label"].map(LABEL2ID)

    train_df, temp_df = train_test_split(
        combined_df, test_size=0.2, stratify=combined_df["label_id"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42
    )

    feedback_test = pd.DataFrame()
    if len(feedback_new) >= 10:
        _, feedback_test = train_test_split(
            feedback_new, test_size=0.2, stratify=feedback_new["label"], random_state=42
        )
        feedback_test["label_id"] = feedback_test["label"].map(LABEL2ID)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    if len(feedback_test) > 0:
        print(f"  Feedback test set: {len(feedback_test)}")

    # Step 5: Train
    print("\n-- Training --")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = EmailDataset(
        train_df["text"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        weights=train_df["weight"].tolist(),
    )
    val_dataset = EmailDataset(
        val_df["text"].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    # Start from the active model if it exists (transfer learning)
    if ACTIVE_MODEL_DIR.exists() and (ACTIVE_MODEL_DIR / "meta.json").exists():
        print(f"  Fine-tuning from active model: {ACTIVE_MODEL_DIR}")
        model = DistilBertForSequenceClassification.from_pretrained(str(ACTIVE_MODEL_DIR))
    else:
        print(f"  Training from scratch: {MODEL_NAME}")
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
        )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_val_acc = 0.0
    version_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if "weight" in batch:
                weights = batch["weight"].to(device)
                per_sample_loss = torch.nn.functional.cross_entropy(
                    outputs.logits, labels, reduction="none"
                )
                loss = (per_sample_loss * weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {correct/total:.4f}")

        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        val_acc = val_correct / val_total
        avg_loss = total_loss / len(train_loader)

        print(f"  Epoch {epoch+1}: train_loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(version_dir)
            tokenizer.save_pretrained(version_dir)
            print(f"    Saved best model (val_acc={val_acc:.4f})")

    # Step 6: Evaluate
    print("\n-- Evaluation --")

    model = DistilBertForSequenceClassification.from_pretrained(str(version_dir))
    model.to(device)
    model.eval()

    main_metrics = _evaluate_model(model, tokenizer, test_df, device, LABELS)
    print(f"  Main test: accuracy={main_metrics['accuracy']}, macro_f1={main_metrics['macro_f1']}")
    for l in LABELS:
        m = main_metrics["per_class"].get(l, {})
        print(f"    {l:20s}: P={m.get('precision', 0):.3f} R={m.get('recall', 0):.3f} F1={m.get('f1', 0):.3f}")

    feedback_metrics = {}
    if len(feedback_test) > 0:
        feedback_metrics = _evaluate_model(model, tokenizer, feedback_test, device, LABELS)
        print(f"\n  Feedback test: accuracy={feedback_metrics['accuracy']}, macro_f1={feedback_metrics['macro_f1']}")

    # Safety net rate on test set
    print("\n  Measuring approval safety net rate on test set...")
    safety_net_stats = _measure_safety_net_rate(model, tokenizer, test_df, device, LABELS)
    print(f"  Safety net fired: {safety_net_stats['flagged']}/{safety_net_stats['total']} ({safety_net_stats['rate']}%)")
    if safety_net_stats["rate"] <= SAFETY_NET_MAX_PCT:
        print(f"    Below {SAFETY_NET_MAX_PCT}% threshold -- model handles approvals well.")
    else:
        print(f"    Above {SAFETY_NET_MAX_PCT}% threshold -- model still misses some approvals.")

    prev_metrics = {}
    if ACTIVE_MODEL_DIR.exists() and (ACTIVE_MODEL_DIR / "meta.json").exists():
        print("\n  Comparing with active model...")
        prev_model = DistilBertForSequenceClassification.from_pretrained(str(ACTIVE_MODEL_DIR))
        prev_model.to(device)
        prev_metrics = _evaluate_model(prev_model, tokenizer, test_df, device, LABELS)
        print(f"  Active model: accuracy={prev_metrics['accuracy']}, macro_f1={prev_metrics['macro_f1']}")

        delta_acc = main_metrics["accuracy"] - prev_metrics["accuracy"]
        delta_f1 = main_metrics["macro_f1"] - prev_metrics["macro_f1"]
        print(f"  Delta: accuracy={delta_acc:+.4f}, macro_f1={delta_f1:+.4f}")

        del prev_model

    # Step 7: Save meta.json
    active_meta = {}
    if (ACTIVE_MODEL_DIR / "meta.json").exists():
        with open(ACTIVE_MODEL_DIR / "meta.json") as f:
            active_meta = json.load(f)

    meta = {
        "labels": LABELS,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "max_length": MAX_LENGTH,
        "model_name": MODEL_NAME,
        "version": version_str,
        "parent_version": active_meta.get("version", "v1"),
        "best_val_acc": round(best_val_acc, 4),
        "epochs": epochs,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "feedback_rows": len(feedback_new),
        "original_rows": len(original_df),
        "provenance": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feedback_cutoff": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
        },
        "evaluation": {
            "main_test": main_metrics,
            "feedback_test": feedback_metrics,
            "previous_model": prev_metrics,
            "safety_net": safety_net_stats,
        },
    }

    with open(version_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    test_df.to_csv(version_dir / "test_set.csv", index=False)
    if len(feedback_test) > 0:
        feedback_test.to_csv(version_dir / "feedback_test_set.csv", index=False)

    print(f"\n  Model saved to: {version_dir}")
    print(f"  Version: {version_str}")

    # Step 8: Promote
    should_promote = False
    safety_ok = safety_net_stats["rate"] <= SAFETY_NET_MAX_PCT

    if promote:
        if not safety_ok:
            print(f"\n  Safety net rate ({safety_net_stats['rate']}%) exceeds {SAFETY_NET_MAX_PCT}% -- blocking promotion.")
            print(f"    Model still misses approvals that the evidence scanner catches.")
            print(f"    Flagged {safety_net_stats['flagged']}/{safety_net_stats['total']} test samples.")
        elif prev_metrics:
            improved = (
                main_metrics["accuracy"] >= prev_metrics["accuracy"]
                and main_metrics["macro_f1"] >= prev_metrics["macro_f1"]
            )
            if improved:
                print(f"\n  Metrics improved + safety net OK ({safety_net_stats['rate']}%) -- promoting {version_str} to active.")
                should_promote = True
            else:
                print(f"\n  Metrics did NOT improve -- skipping promotion.")
                print(f"    New:  acc={main_metrics['accuracy']}, f1={main_metrics['macro_f1']}")
                print(f"    Prev: acc={prev_metrics['accuracy']}, f1={prev_metrics['macro_f1']}")
        else:
            print(f"\n  No previous model to compare + safety net OK ({safety_net_stats['rate']}%) -- promoting {version_str}.")
            should_promote = True

    if should_promote:
        if ACTIVE_MODEL_DIR.exists():
            backup_dir = MODELS_DIR / f"email_classifier_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(ACTIVE_MODEL_DIR, backup_dir)
            print(f"  Backed up active model to: {backup_dir}")

        for f in version_dir.iterdir():
            shutil.copy2(f, ACTIVE_MODEL_DIR / f.name)
        print(f"  Promoted {version_str} to {ACTIVE_MODEL_DIR}")

    # Step 9: Mark feedback as consumed
    if "feedback_id" in feedback_new.columns:
        feedback_ids = feedback_new["feedback_id"].dropna().astype(int).tolist()
        _mark_feedback_trained(feedback_ids, version_str)
        print(f"  Marked {len(feedback_ids)} feedback rows as used_in_training=true")

    print(f"\n{'=' * 70}")
    print(f"Retraining complete: {version_str}")
    print(f"{'=' * 70}")

    return {
        "version": version_str,
        "version_dir": str(version_dir),
        "metrics": main_metrics,
        "feedback_metrics": feedback_metrics,
        "promoted": should_promote,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain DistilBERT with feedback data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--promote", action="store_true", help="Auto-promote if metrics improve")
    parser.add_argument("--dry-run", action="store_true", help="Show data stats only")
    args = parser.parse_args()

    retrain(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        promote=args.promote,
        dry_run=args.dry_run,
    )
