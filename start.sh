#!/bin/bash

# Start FastAPI backend in the background on port 8000
echo "🚀 Starting FastAPI AutoGluon Backend Engine..."
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait briefly for backend processes to spin up initialization
sleep 5

# Start Streamlit frontend on port 7860 (Hugging Face Default)
echo "🖥️ Starting Streamlit Frontend Dashboard..."
uv run streamlit run dashboard/app.py --server.port 7860 --server.address 0.0.0.0