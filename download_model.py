#!/usr/bin/env python3
"""
Download model weights from Supabase Storage bucket on startup.

Used when deploying to Railway without baking weights into the Docker image.
Expects model files in Supabase Storage bucket 'ml-models' under path
'email_classifier/{version}/'.

Set env vars:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE (or SUPABASE_SERVICE_KEY)
  MODEL_VERSION (optional, defaults to "v1")
"""

import os
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models" / "email_classifier"
BUCKET = "ml-models"

# Files that make up a DistilBERT model
MODEL_FILES = [
    "config.json",
    "meta.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def download():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        print("SUPABASE_URL and SUPABASE_SERVICE_ROLE required for model download")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(url, key)

    version = os.environ.get("MODEL_VERSION", "v1")
    remote_prefix = f"email_classifier/{version}"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        remote_path = f"{remote_prefix}/{filename}"
        local_path = MODEL_DIR / filename

        if local_path.exists():
            print(f"  {filename} already exists, skipping")
            continue

        print(f"  Downloading {remote_path}...")
        try:
            data = sb.storage.from_(BUCKET).download(remote_path)
            with open(local_path, "wb") as f:
                f.write(data)
            size_mb = len(data) / 1024 / 1024
            print(f"    Saved {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"    ERROR downloading {filename}: {e}")
            sys.exit(1)

    print(f"\nModel downloaded to {MODEL_DIR}")


if __name__ == "__main__":
    download()
