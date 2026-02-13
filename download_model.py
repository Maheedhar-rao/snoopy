#!/usr/bin/env python3
"""
Download model weights from GitHub Releases on startup.

Used when deploying to Railway — tokenizer files are baked into the Docker
image but model.safetensors (255MB) is fetched from a GitHub Release asset.

Set env vars:
  GITHUB_TOKEN          — PAT with repo read access (for private repos)
  MODEL_VERSION         — release tag (default "v1")
  GITHUB_REPO           — owner/repo (default "Maheedhar-rao/snoopy")
"""

import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_DIR = Path(__file__).resolve().parent / "models" / "email_classifier"
WEIGHT_FILE = "model.safetensors"


def download():
    token = os.environ.get("GITHUB_TOKEN")
    version = os.environ.get("MODEL_VERSION", "v1")
    repo = os.environ.get("GITHUB_REPO", "Maheedhar-rao/snoopy")

    local_path = MODEL_DIR / WEIGHT_FILE

    if local_path.exists():
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  {WEIGHT_FILE} already exists ({size_mb:.1f} MB), skipping")
        return

    # GitHub Release asset download URL
    asset_url = f"https://github.com/{repo}/releases/download/{version}/{WEIGHT_FILE}"
    print(f"  Downloading {WEIGHT_FILE} from release {version}...")
    print(f"  URL: {asset_url}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        req = Request(asset_url)
        req.add_header("Accept", "application/octet-stream")
        if token:
            req.add_header("Authorization", f"token {token}")

        with urlopen(req, timeout=300) as resp:
            data = resp.read()

        with open(local_path, "wb") as f:
            f.write(data)

        size_mb = len(data) / 1024 / 1024
        print(f"  Saved {WEIGHT_FILE} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ERROR downloading {WEIGHT_FILE}: {e}")
        sys.exit(1)

    print(f"\nModel ready at {MODEL_DIR}")


if __name__ == "__main__":
    download()
