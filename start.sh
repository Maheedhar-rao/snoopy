#!/bin/bash
set -e

echo "Starting ML classification service..."
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
