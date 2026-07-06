from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.schemas import MSMEEvaluationResponse
from backend.credit_engine import evaluate_credit_profile, get_all_msmes, init_model

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", status_code=status.HTTP_200_OK)
def system_health_check():
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