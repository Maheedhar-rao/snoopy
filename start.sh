#!/bin/bash
set -e

MODEL_DIR="/app/models/email_classifier"

# Download model weights from GitHub Release if not baked into image
# Both API and monitor need the BERT model
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Model weights not found. Downloading from GitHub Release..."
    python /app/download_model.py
fi

# Monitor mode: set MONITOR_WEB_MODE=1 to run the live monitor instead of the API
if [ "$MONITOR_WEB_MODE" = "1" ]; then
    echo "Starting live monitor service..."
    exec uvicorn live_monitor:web_app --host 0.0.0.0 --port ${PORT:-8080}
fi

echo "Starting ML classification service..."
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
