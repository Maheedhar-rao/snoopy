#!/usr/bin/env python3
"""Upload model.safetensors to Supabase Storage using TUS resumable upload.

Handles files >50MB which exceed Supabase's standard upload limit.
"""

import base64
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

BUCKET = "ml-models"
LOCAL_FILE = Path(__file__).resolve().parent / "models" / "email_classifier" / "model.safetensors"
REMOTE_PATH = "email_classifier/v1/model.safetensors"
CHUNK_SIZE = 6 * 1024 * 1024  # 6MB chunks


def upload():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE env vars first.")
        sys.exit(1)

    if not LOCAL_FILE.exists():
        print(f"Model file not found: {LOCAL_FILE}")
        sys.exit(1)

    file_size = LOCAL_FILE.stat().st_size
    size_mb = file_size / 1024 / 1024
    print(f"Uploading {LOCAL_FILE.name} ({size_mb:.1f} MB) via TUS resumable upload...")

    storage_url = f"{url}/storage/v1"
    tus_url = f"{storage_url}/upload/resumable"

    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
    }

    # Step 1: Create the upload
    upload_metadata = base64.b64encode(
        json.dumps({
            "bucketName": BUCKET,
            "objectName": REMOTE_PATH,
            "contentType": "application/octet-stream",
        }).encode()
    ).decode()

    create_headers = {
        **headers,
        "Tus-Resumable": "1.0.0",
        "Upload-Length": str(file_size),
        "Upload-Metadata": f"bucketName {base64.b64encode(BUCKET.encode()).decode()},objectName {base64.b64encode(REMOTE_PATH.encode()).decode()},contentType {base64.b64encode(b'application/octet-stream').decode()}",
    }

    print("Creating resumable upload session...")
    with httpx.Client(timeout=30) as client:
        resp = client.post(tus_url, headers=create_headers, content=b"")

    if resp.status_code not in (200, 201):
        print(f"Failed to create upload: {resp.status_code} {resp.text}")
        sys.exit(1)

    upload_url = resp.headers.get("Location")
    if not upload_url:
        print(f"No Location header in response. Headers: {dict(resp.headers)}")
        sys.exit(1)

    print(f"Upload session created. Sending {file_size // CHUNK_SIZE + 1} chunks...")

    # Step 2: Upload chunks
    offset = 0
    with open(LOCAL_FILE, "rb") as f:
        with httpx.Client(timeout=120) as client:
            while offset < file_size:
                chunk = f.read(CHUNK_SIZE)
                chunk_len = len(chunk)

                patch_headers = {
                    **headers,
                    "Tus-Resumable": "1.0.0",
                    "Upload-Offset": str(offset),
                    "Content-Type": "application/offset+octet-stream",
                    "Content-Length": str(chunk_len),
                }

                resp = client.patch(upload_url, headers=patch_headers, content=chunk)

                if resp.status_code not in (200, 204):
                    print(f"Chunk upload failed at offset {offset}: {resp.status_code} {resp.text}")
                    sys.exit(1)

                offset += chunk_len
                pct = offset / file_size * 100
                print(f"  {pct:5.1f}% ({offset / 1024 / 1024:.1f} / {size_mb:.1f} MB)")

    print(f"\nUploaded successfully to {BUCKET}/{REMOTE_PATH}")


if __name__ == "__main__":
    upload()
