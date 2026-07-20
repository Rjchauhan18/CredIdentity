import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import streamlit as st

# Optimized loopback adapter binding for high-speed intra-container networking
API_URL = "http://127.0.0.1:8000/api/v1"

# --- Senior Professional Architecture: Connection Pooling & Resilient Retries ---
http_session = requests.Session()
retry_strategy = Retry(
    total=3,                          # Max internal retries before escalating
    backoff_factor=0.5,               # Exponential delay spacing: 0.5s, 1.0s, 2.0s
    status_forcelist=[502, 503, 504], # Recoverable transient cloud gateways
    raise_on_status=False             # Allow local code block to evaluate response statuses
)
http_session.mount("http://", HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10))
http_session.mount("https://", HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10))


class DropdownOptions(list):
    """
    A professional sequence wrapper extending native lists.
    Safely bridges structural metadata parameters directly to the front-end components
    without breaking standard iterable rendering requirements.
    """
    def __init__(self, *args, error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.error = error or {"message": "Unknown API communication failure."}


@st.cache_data
def get_msme_data(msme_id):
    """
    Fetches live ML engine predictions directly from the backend API.
    Cached indefinitely per unique MSME ID to prevent unneeded heavy AutoGluon evaluations.
    """
    msme_id = str(msme_id).strip()
    
    try:
        # A 20-second threshold allows multi-layer stacking evaluations to complete safely
        response = http_session.get(f"{API_URL}/evaluate/{msme_id}", timeout=20)
        
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


def evaluate_raw_profile(raw_json: dict):
    """POST a raw AA/GST/EPFO profile to the backend and return the scored result.

    Not cached — each pasted profile is a one-off. Returns (data, error_message).
    """
    try:
        response = http_session.post(f"{API_URL}/evaluate-raw", json=raw_json, timeout=20)
        if response.status_code == 200:
            return response.json(), None
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"Backend rejected the profile (HTTP {response.status_code}): {detail}"
    except requests.exceptions.Timeout:
        return None, "Connection timed out while scoring the raw profile."
    except requests.exceptions.RequestException:
        return None, "Unable to reach the backend scoring engine."


@st.cache_data
def get_validation_report():
    """Fetch the precomputed offline validation + backtest artifact. Cached for the session."""
    try:
        response = http_session.get(f"{API_URL}/validation-report", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


@st.cache_data
def get_msme_dropdown_options():
    """
    Fetches the actual list of companies from the API once to populate selection elements.
    Cached permanently for the session runtime to completely eliminate background log spam.
    """
    try:
        response = http_session.get(f"{API_URL}/msmes", timeout=5)
        if response.status_code == 200:
            data = response.json()
            raw_list = [f"{item['company_name']} ({item['msme_id']})" for item in data.get('msmes', [])]
            return DropdownOptions(raw_list)
            
    except requests.exceptions.RequestException as e:
        # Log error cleanly into terminal environment while surfacing structural routing flags
        print(f"[API CLIENT WARNING] Fallback triggered via connection management layer: {str(e)}")
    
    # Return an empty list wrapper so 'if options:' safely branches to 'else:' inside the UI layer
    return DropdownOptions([], error={"message": "Backend engine is currently initializing or unreachable."})