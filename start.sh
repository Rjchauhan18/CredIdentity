#!/bin/bash

echo "🚀 [1/3] Starting FastAPI AutoGluon Backend Engine..."
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "⏳ [2/3] Waiting for Backend to download weights and boot (Checking every 5 seconds)..."

# Initialize a counter
ATTEMPTS=0
MAX_ATTEMPTS=300 # 300 attempts * 5 seconds = 25 minutes max wait time
                 # (heavy AutoGluon ensemble: first cold start downloads 400+ files
                 #  AND loads the full stack into memory, which exceeds 10 min)

# Ping the readiness endpoint until the model is loaded (returns 200; 503 while loading)
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' http://127.0.0.1:8000/health)" != "200" ]]; do
    ATTEMPTS=$((ATTEMPTS+1))
    if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
        echo "❌ Backend failed to become ready within 25 minutes. Check the logs above."
        kill $BACKEND_PID
        exit 1
    fi
    echo "   ...still loading (Attempt $ATTEMPTS/$MAX_ATTEMPTS). Sleeping for 5s."
    sleep 5
done

echo "✅ [3/3] Backend is fully online! Starting Streamlit Frontend..."
uv run streamlit run dashboard/app.py --server.port 7860 --server.address 0.0.0.0