import json
import os

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.schemas import MSMEEvaluationResponse, RawMSMEProfile
from backend.credit_engine import (
    evaluate_credit_profile,
    evaluate_raw_profile,
    get_all_msmes,
    init_model,
    is_ready,
)

# Lifespan manager to load the heavy AutoGluon model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Starting up API and initializing Model Engine...")
    init_model()
    yield
    print("🛑 Shutting down API and Model Engine...")

app = FastAPI(
    title="IDBI CredIdentity Backend Scorer Engine",
    description="Algorithmic Underwriting Network for MSME Financial Footprint Ingestion",
    version="2.0.0",
    lifespan=lifespan
)

# The backend is only reached over the container loopback by the Streamlit frontend,
# so keep CORS tight. allow_origins=["*"] with allow_credentials=True is an invalid combo
# browsers reject anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:7860", "http://localhost:7860"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def system_health_check():
    """Readiness probe.

    The heavy model is loaded synchronously in the lifespan startup, so uvicorn does
    not accept connections until it is ready. In practice the boot poller (start.sh)
    therefore sees connection-refused during load and a 200 once the server is up. The
    503 branch below is a defensive guard for any future move to background loading;
    it keeps the endpoint correct either way.
    """
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model engine is still initializing."
        )
    return {"status": "operational", "engine": "Cloud AutoGluon Ensemble Active"}

@app.get("/api/v1/evaluate/{msme_id}", response_model=MSMEEvaluationResponse)
def evaluate_msme_profile(msme_id: str):
    evaluation_result = evaluate_credit_profile(msme_id)
    if not evaluation_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MSME Identity context matching identifier code '{msme_id}' could not be located."
        )
    return evaluation_result

@app.get("/api/v1/msmes")
def list_msmes():
    """Endpoint to populate the frontend search dropdown"""
    return {"msmes": get_all_msmes()}


# Cap raw-profile uploads so a malformed/huge payload can't exhaust memory.
MAX_RAW_PROFILE_BYTES = 2 * 1024 * 1024  # 2 MB


@app.post("/api/v1/evaluate-raw", response_model=MSMEEvaluationResponse)
async def evaluate_raw(request: Request):
    """Score a raw AA/GST/EPFO/UPI profile on the fly.

    SECURITY NOTE: this endpoint is UNAUTHENTICATED. It is safe in the current
    deployment only because CORS + the container network restrict it to the
    co-located Streamlit frontend on loopback. Any internet-facing deployment MUST
    put authentication (API key / OAuth) and rate limiting in front of it — it runs
    model inference on arbitrary caller input. Body size is capped below.
    """
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model engine is still initializing.",
        )

    body = await request.body()
    if len(body) > MAX_RAW_PROFILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Raw profile exceeds {MAX_RAW_PROFILE_BYTES // (1024*1024)} MB limit.",
        )
    try:
        payload = json.loads(body)
        profile = RawMSMEProfile(**payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid raw MSME profile: {exc}",
        )

    try:
        return evaluate_raw_profile(profile.model_dump())
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not derive features from profile (missing/invalid field): {exc}",
        )


@app.get("/api/v1/validation-report")
def validation_report():
    """Serve the precomputed offline validation + backtest artifact (if present)."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/validation_report.json"))
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Validation report not generated yet. Run scripts/build_validation_report.py.",
        )
    with open(path) as f:
        return json.load(f)