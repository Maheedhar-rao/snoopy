#!/usr/bin/env python3
"""
Fine-tune DistilBERT for email response classification.

Usage:
    python train.py                          # train with defaults
    python train.py --epochs 5 --batch-size 32
    python train.py --data data/email_responses.csv
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "email_responses.csv"
MODEL_DIR = PROJECT_DIR / "models" / "email_classifier"

# Label mapping
LABELS = ["APPROVED", "DECLINED", "OTHER", "STIPS_REQUIRED"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MAX_LENGTH = 512
MODEL_NAME = "distilbert-base-uncased"


class EmailDataset(Dataset):
    """Lazy tokenization — tokenizes per __getitem__ to avoid memory blowup."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
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
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train(
    data_path: str = str(DATA_PATH),
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slower)")

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    print(f"  Total samples: {len(df)}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}\n")

    # Encode labels
    df["label_id"] = df["label"].map(LABEL2ID)

    # Split: train / val / test (stratified)
    train_df, temp_df = train_test_split(
        df, test_size=val_split + test_split, stratify=df["label_id"], random_state=42
    )
    relative_test = test_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df["label_id"], random_state=42
    )

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save test set for evaluation
    test_path = PROJECT_DIR / "data" / "test_set.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  Test set saved to: {test_path}")

    # Tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Datasets (lazy tokenization — no memory blowup)
    print("Creating datasets (lazy tokenization)...")
    train_dataset = EmailDataset(
        train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer
    )
    val_dataset = EmailDataset(
        val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    # Model
    print(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # Training loop
    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training: {epochs} epochs, {len(train_loader)} batches/epoch\n")

    for epoch in range(epochs):
        # Train
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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 25 == 0:
                running_acc = correct / total
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {running_acc:.4f}")

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"\n  Epoch {epoch+1}/{epochs} complete:")
        print(f"    Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"    Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"    Saved best model (val_acc={val_acc:.4f})\n")

    # Save label mapping
    meta = {
        "labels": LABELS,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "best_val_acc": best_val_acc,
        "epochs": epochs,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT email classifier")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
