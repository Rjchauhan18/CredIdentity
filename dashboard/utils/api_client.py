import json
import os
import requests
import streamlit as st

# This is the URL where Member 1's FastAPI server will eventually live
API_URL = "http://localhost:8000/api/v1/evaluate"

@st.cache_data(ttl=60)
def get_msme_data(msme_id="MSME-2026-X94B"):
    """
    Attempts to fetch live ML predictions from the backend API.
    If the API is unreachable (e.g., Member 1 is still coding), 
    it falls back to the static mock contract safely.
    """
    try:
        # Attempt to contact Member 1's Machine Learning API
        response = requests.get(f"{API_URL}/{msme_id}", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass # API is down, fallback triggered
        
    # Fallback Mechanism: Load local mock data
    current_dir = os.path.dirname(__file__)
    mock_path = os.path.join(current_dir, '..', 'mock_data', 'mock_msme_response.json')
    
    with open(mock_path, "r") as f:
        return json.load(f)