#!/usr/bin/env python3
"""
Evaluate the trained DistilBERT email classifier on the held-out test set.

Usage:
    python evaluate.py                    # evaluate on saved test set
    python evaluate.py --data path/to.csv # evaluate on custom data
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "email_classifier"
TEST_PATH = PROJECT_DIR / "data" / "test_set.csv"


def evaluate(data_path: str = str(TEST_PATH), show_errors: int = 10):
    # Load meta
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        print(f"No model found at {MODEL_DIR}. Run train.py first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    labels = meta["labels"]
    max_length = meta.get("max_length", 512)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model + tokenizer
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # Load test data
    print(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df[df["label"].isin(labels)].reset_index(drop=True)
    print(f"  Test samples: {len(df)}")

    # Predict in batches
    all_preds = []
    all_probs = []
    batch_size = 32

    for i in range(0, len(df), batch_size):
        batch_texts = df["text"].iloc[i : i + batch_size].tolist()

        encodings = tokenizer(
            batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    # Map predictions back to labels
    df["predicted"] = [labels[p] for p in all_preds]
    df["confidence"] = [max(p) for p in all_probs]
    df["correct"] = df["label"] == df["predicted"]

    # Classification report
    true_labels = df["label"].tolist()
    pred_labels = df["predicted"].tolist()

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(true_labels, pred_labels, labels=labels, digits=4))

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    print("CONFUSION MATRIX")
    print("-" * 60)

    header = f"{'':20s}" + "".join(f"{l:>15s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:20s}" + "".join(f"{cm[i][j]:>15d}" for j in range(len(labels)))
        print(row)

    # Overall accuracy
    accuracy = df["correct"].mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({df['correct'].sum()}/{len(df)})")

    # Confidence analysis
    print(f"\n{'=' * 60}")
    print("CONFIDENCE ANALYSIS")
    print(f"{'=' * 60}")

    for threshold in [0.5, 0.7, 0.85, 0.9, 0.95]:
        above = df[df["confidence"] >= threshold]
        if len(above) > 0:
            acc = above["correct"].mean()
            coverage = len(above) / len(df)
            print(f"  Threshold >= {threshold:.2f}: accuracy={acc:.4f}, coverage={coverage:.1%} ({len(above)}/{len(df)})")

    # Show worst errors
    if show_errors > 0:
        errors = df[~df["correct"]].sort_values("confidence", ascending=False).head(show_errors)
        if len(errors) > 0:
            print(f"\n{'=' * 60}")
            print(f"TOP {len(errors)} HIGH-CONFIDENCE ERRORS (needs review)")
            print(f"{'=' * 60}")
            for _, row in errors.iterrows():
                text_preview = row["text"][:100].replace("\n", " ")
                print(f"\n  True: {row['label']:20s} | Predicted: {row['predicted']:20s} | Conf: {row['confidence']:.3f}")
                print(f"  Text: {text_preview}...")

    # Save predictions
    out_path = PROJECT_DIR / "data" / "test_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT email classifier")
    parser.add_argument("--data", type=str, default=str(TEST_PATH))
    parser.add_argument("--show-errors", type=int, default=10)
    args = parser.parse_args()

    evaluate(data_path=args.data, show_errors=args.show_errors)
