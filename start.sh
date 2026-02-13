#!/bin/bash
set -e

MODEL_DIR="/app/models/email_classifier"

# Download model weights from GitHub Release if not baked into image
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Model weights not found. Downloading from GitHub Release..."
    python /app/download_model.py
fi

echo "Starting ML classification service..."
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
