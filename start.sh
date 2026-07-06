#!/bin/bash

echo "🚀 [1/3] Starting FastAPI AutoGluon Backend Engine..."
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "⏳ [2/3] Waiting for Backend to download weights and boot (Checking every 5 seconds)..."

# Initialize a counter
ATTEMPTS=0
MAX_ATTEMPTS=120 # 120 attempts * 5 seconds = 10 minutes max wait time

# Ping the health check endpoint until it returns HTTP 200
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' http://127.0.0.1:8000/docs)" != "200" ]]; do
    ATTEMPTS=$((ATTEMPTS+1))
    if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
        echo "❌ Backend failed to start within 10 minutes. Check the logs."
        kill $BACKEND_PID
        exit 1
    fi
    echo "   ...still loading (Attempt $ATTEMPTS/$MAX_ATTEMPTS). Sleeping for 5s."
    sleep 5
done

echo "✅ [3/3] Backend is fully online! Starting Streamlit Frontend..."
uv run streamlit run dashboard/app.py --server.port 7860 --server.address 0.0.0.0