#!/bin/bash
set -e

MODEL_DIR="/app/models/email_classifier"

# If model weights are missing, download from Supabase Storage
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Model not found locally. Downloading from Supabase Storage..."
    python /app/download_model.py
fi

echo "Starting ML classification service..."
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
