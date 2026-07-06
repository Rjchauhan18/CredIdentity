import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1"

@st.cache_data
def get_msme_data(msme_id):
    """
    Fetches live ML engine predictions directly from the backend API.
    Cached indefinitely per unique MSME ID to prevent unneeded heavy AutoGluon evaluations.
    """
    msme_id = str(msme_id).strip()
    
    try:
        # 20-second threshold allows complex multi-layer stacking layers to complete safely
        response = requests.get(f"{API_URL}/evaluate/{msme_id}", timeout=20)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error(f"🔍 MSME ID '{msme_id}' could not be located in your CSV dataset.")
        elif response.status_code == 422:
            st.error("❌ Schema Validation Error: The data structure sent doesn't match the Pydantic model.")
        else:
            st.error(f"💥 Backend Error (Status {response.status_code}): Check your FastAPI terminal console for the traceback.")
            
    except requests.exceptions.Timeout:
        st.error("⏳ Connection Timed Out! The AutoGluon model is taking longer than 20 seconds to compute.")
    except requests.exceptions.RequestException:
        st.error("🔌 Unable to connect to the Backend Scoring Engine server. Verify main.py is active.")
        
    return None

@st.cache_data
def get_msme_dropdown_options():
    """
    Fetches the actual list of companies from the API once to populate selection elements.
    Cached permanently for the session runtime to completely eliminate background log spam.
    """
    try:
        response = requests.get(f"{API_URL}/msmes", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [f"{item['company_name']} ({item['msme_id']})" for item in data.get('msmes', [])]
    except requests.exceptions.RequestException:
        pass
    
    return ["No MSME records found. Ensure backend API is running."]