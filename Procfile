web: uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
monitor: MONITOR_WEB_MODE=1 uvicorn live_monitor:web_app --host 0.0.0.0 --port ${PORT:-8081}
