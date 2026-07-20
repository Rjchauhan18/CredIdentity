---
title: IDBI MSME CredIdentity
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🏦 IDBI CredIdentity: AI-Driven MSME Financial Health Card

[![FastAPI badge reading API FastAPI, representing the backend API framework used in the project](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit badge reading Frontend Streamlit, representing the frontend visualization framework used in the project](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![AutoGluon badge reading ML Engine AutoGluon, representing the machine learning engine used](https://img.shields.io/badge/ML_Engine-AutoGluon-orange?style=flat)]()
[![Package manager badge reading Manager uv, representing the uv package manager used for dependency management](https://img.shields.io/badge/Manager-uv-blueviolet?style=flat)](https://github.com/astral-sh/uv)
[![GitHub Actions badge indicating HF Sync workflow status for repository automation](https://github.com/Rjchauhan18/CredIdentity/actions/workflows/hf_sync.yml/badge.svg)](https://github.com/Rjchauhan18/CredIdentity/actions)
> **"Less than 11% of Indian MSMEs have access to formal credit — not because they are bad businesses, but because traditional scoring systems look at the wrong metrics. We fix that by creating a credit identity starting from zero."**

### 🎯 **Live Demo Platform:** [🔗 Explore the Live Application on Hugging Face Spaces](https://huggingface.co/spaces/Rjchauhan/credidentity)
### 👉 Click Here to Open : [![Kaggle badge reading Kaggle Notebook, linking to the project notebook on Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/code/chauhanrj/idbi-msme-financial-health-score)

---

## 🎯 Hackathon Track
* **Track:** Track 03: MSME Lending — AI-Driven Financial Health Card
* **Theme:** Financial Inclusion, Digital Lending, and Credit Decisioning
* **Problem Statement:** Traditional credit evaluation relies heavily on collateral and historical CIBIL data. New-to-Credit (NTC) and New-to-Bank (NTB) micro-enterprises are systematically excluded due to the absence of a unified assessment framework, leading to high rejection rates.
* **Our Solution:** **CredIdentity** bridges India's ₹25 trillion MSME credit gap. By legally aggregating consent-based alternative digital public infrastructure footprints (GST, UPI, AA, and EPFO), the platform runs a real-time predictive pipeline to generate an auditable, multidimensional Financial Health Score mapped directly to eligible OCEN 4.0 lending products.

---

## 🏗️ Repository Architecture & Layout

The project is cleanly modularized into a data pipeline, a machine learning backend API container, and a production-ready Streamlit analytical application dashboard.

```text
.
├── backend/                        # Production API Layer
│   ├── main.py                     # FastAPI server hosting model loading & endpoints
│   ├── credit_engine.py            # Inference logic & script execution
│   └── schemas.py                  # Pydantic data contract validation objects
├── dashboard/                      # UI Visualization Layer
│   ├── app.py                      # Multi-page Streamlit configuration & sidebar navigation
│   ├── pages/
│   │   ├── health_card.py          # View 1: MSME facing identity health card profile
│   │   ├── loan_officer.py         # View 2: Auditor underwriting desk & risk metrics matrix
│   │   └── portfolio.py            # View 3: Executive macro liquidity distribution overview
│   └── utils/
│       ├── api_client.py           # Core backend interface wired to processed datasets
│       ├── loan_matcher.py         # OCEN 4.0 matching matrix rule engine
│       └── visualizations.py       # Custom Plotly radar and SHAP waterfall chart configurations
├── data/                           # Data Storage Matrix
│   ├── processed/                  # Final pipeline output targets (12 engineered features)
│   │   └── msme_final_engineered_features.csv
│   └── raw/                        # Synthetic input base structures
│       └── raw_msme_data.json
├── data_pipeline/                  # Feature Engineering Suite
│   ├── engineer_features.py        # Generates structured metrics from raw footprints
│   └── generate_data.py            # Synthetic DPI data creation engine (Faker)
├── Dockerfile                      # Unified cloud multi-stage container manifest
├── pyproject.toml                  # Project requirements configuration
├── start.sh                        # Dual-process background container initiator
└── uv.lock                         # Astral-uv deterministic lockfile

```

---

## 🧠 The Four Analytical Pillars & 12 Core Features

Instead of a single opaque number, the machine learning models capture signals across **4 Core Pillars**, breaking down data into **12 targeted operational features**:

| Pillar | Features Engineered | Core Regulatory / Business Metric Measured |
| :--- | :--- | :--- |
| **💧 Cash Flow Resiliency**  | `Average Monthly Balance`<br>`Inflow-to-Outflow Ratio`<br>`Bounce / Mandate Failure Rate`<br>`Balance Depletion Speed`  | Real-time liquidity checking; flags payment distress trends immediately before defaults occur. |
| **📊 Commercial Revenue**  | `Monthly Revenue Growth Rate`<br>`B2B vs B2C Revenue Ratio`  | Verifies real business transactional velocity instead of unverified self-reported claims. |
| **👥 Operational Stability**  | `Active Employee Count Trend`<br>`EPFO Wage Consistency Score`  | Proves the enterprise is actively running using live employee welfare contribution cadence. |
| **✅ Compliance & Risk**  | `GSTR-1 vs GSTR-3B Variance`<br>`Late GST Filing Frequency`<br>`UPI Transaction Velocity`<br>`Night Transaction Ratio`  | Flags tax variations, sudden anomaly volume drops, or fraudulent transaction behaviors early. |

---

## ⚡ Technical Stack & Advanced ML Infrastructure

* **Data Aggregation:** Simulates the Account Aggregator (AA) network, Unified Lending Interface (ULI), and GSTN API data transfers via secure, clean structured JSON formats.
* **Predictive Ensemble (AutoGluon Tabular):** Benches a production-grade multi-layer stacking tabular network combining TabNet sequential attention weights, tuned XGBoost/CatBoost gradient-boosted trees, and unsupervised Isolation Forests for anomaly fraud checking.
* **Explainable AI (XAI):** Built-in explicit local explanations utilizing **SHAP (SHapley Additive exPlanations)**. The system isolates the top 3 strengths and top 3 risks, transforming feature values into transparent, plain English strings for automated loan officer assessment notes.
* **Loan Matching Matrix:** Maps the live health score (300-900 CIBIL calibrated scale) directly into an automated rule filter executing instant pre-approvals on **OCEN 4.0** product registries.

---

## 📈 Model Validation & Dataset Refinement

Developing a robust MSME credit scoring model required more than maximizing predictive performance. During initial experimentation, the synthetic dataset produced near-perfect classification results (ROC-AUC ≈ **0.9999**), indicating that certain engineered compliance features were unintentionally leaking information about the target variable.

Rather than accepting these unrealistic results, the entire synthetic data generation pipeline was audited and redesigned.

### Key Improvements

- Reduced deterministic relationships between engineered features and the default label.
- Introduced overlapping statistical distributions for GST and EPFO behaviour.
- Added realistic stochastic variability to compliance patterns.
- Increased feature overlap between healthy and distressed MSMEs.
- Re-engineered the dataset to better represent real-world uncertainty.

### Performance Comparison

| Metric | Initial Dataset | Final Dataset |
|---------|---------------:|--------------:|
| ROC-AUC | **0.9999** | **0.9777** |
| Accuracy | **99%+** | **94.60%** |
| Balanced Accuracy | — | **87.65%** |
| MCC | — | **0.7818** |

### Final Model Characteristics

The final AutoGluon ensemble no longer relies on a single dominant feature. Instead, predictions are driven by multiple complementary financial indicators, including:

- Average Monthly Balance
- EPFO Wage Consistency
- GST Filing Behaviour
- Cash Flow Ratio
- Balance Depletion Speed

SHAP analysis confirms that the model bases its decisions on economically meaningful signals rather than synthetic shortcuts, resulting in a substantially more realistic MSME financial health assessment pipeline.

---

## 🪶 Model Slimming: 10 GB → 43 MB (99.6% Smaller)

The trained AutoGluon ensemble was accurate but enormous — a **34-model, multi-layer stack** where every model carried 3 bagging fold-copies plus the full training cache. Serving predictions needs none of that weight.

We built a slim deployment model that keeps a **7-model `WeightedEnsemble_L2_FULL` blend** (the refit `_FULL` variant, which collapses each model's fold-copies into a single trained booster). That one change is where virtually all of the size lived.

| Metric | Full Model | Slim Model | Reduction |
| ------ | ---------: | ---------: | --------: |
| Footprint | 10,077 MB | 43.5 MB | **99.57%** (232× smaller) |
| File count | 1,231 | 33 | 97.3% |
| Cold-start load | tens of seconds | ~seconds | — |

The size reduction cost almost nothing in accuracy, and this was **verified company-by-company**, not assumed:

- **Max credit-score drift: ≤5 points out of 900** (0.56% worst case).
- **23 of 50 companies score identically**; the single worst case moves 368 → 373.
- **Zero tier changes** across all 50 companies — no lending decision flips.

Fidelity was confirmed end-to-end by serving the slim model through the real backend on a clean download from the public Hugging Face repo, then matching every endpoint score against the offline drift report.

> **Scope note:** this shrinks the *model weights* 232×, not the runtime container image (Python + PyTorch + AutoGluon + CUDA libraries), which is unchanged.

### How the slim model is built

[`scripts/slim_model.py`](scripts/slim_model.py) is **non-destructive** — it copies the source model, trims the copy, proves the scores on the held-out demo cohort, and leaves the original untouched. Re-uploading the slim copy to the HF repo is a deliberate manual step.

The core trick is the **`_FULL` refit variant**. A bagged model (e.g. `LightGBM_r96_BAG_L1`) delegates prediction to per-fold children (`S1F1/S1F2/S1F3`) stored *inside* the bag's own folder, not as standalone trainer entries. Ripped out on its own, the bag asks the trainer to load `S1F1`, gets "model does not exist", and crashes with `NoneType.predict` — so **a bagged model cannot serve in isolation**. The fix is to serve the non-bagged `_FULL` refit (a single booster trained on all data, no fold children), which AutoGluon builds during training or on demand via `refit_full()`. The script supports several keep-modes (`best`, `single`, `ensemble_l2`, `ensemble_l3`, `ensemble_topn`); the deployed model uses the `WeightedEnsemble_L2_FULL` blend.

### Model versioning & environment pins

The slim model is a **serialized AutoGluon artifact**, so it is coupled to the toolchain it was trained with. These pins live in the model repo's `metadata.json` and matter because AutoGluon asserts on a mismatch at load time:

| Component | Trained/pinned version | Why it matters |
| --------- | ---------------------- | -------------- |
| AutoGluon | `1.5.0` | Serialization format is version-specific; loading under a different major/minor emits metadata-mismatch warnings and can fail. |
| Python | `3.12.13` (`3.12` series) | The FastAI/Torch fold learners pickle `PosixPath` objects; loading under a *different* Python + OS can break unpickling. |
| OS | `Linux` | Model was trained on Linux; the deploy target (HF Spaces Docker) is also Linux, so paths deserialize cleanly. |
| PyTorch | `2.9.1` | Backs the neural-net base learners. |
| LightGBM / XGBoost / CatBoost | `4.6.0` / `3.1.3` / `1.2.10` | Tree boosters in the ensemble; each deserializes its own booster format. |
| scikit-learn / NumPy / pandas | `1.6.1` / `2.0.2` / `2.3.3` | Feature pipeline + RandomForest/ExtraTrees learners. |

> **Local-dev caveat:** loading this model on **Windows or a non-3.12 Python** trips a `NotImplementedError: cannot instantiate 'PosixPath'` from the pickled FastAI learners. Production is unaffected (Linux + matching Python). For local inspection only, a `pathlib.PosixPath = pathlib.WindowsPath` shim works around it — never commit that shim into serving code. To regenerate the model cleanly, match the versions above (`uv sync` installs the locked set).

Whenever you retrain or bump AutoGluon, re-run `scripts/slim_model.py` to rebuild the slim artifact against the new toolchain and re-verify the per-company drift before re-uploading — a slim model built under one AutoGluon version should not be served under another.

---

## 🛠️ Local Installation & Running Guide

Ensure you have Python 3.10 installed. This project leverages the ultra-fast package manager `uv` by Astral for execution safety.
## Run using docker and bash script

Alternatively, make the global shell script executable to spin up both systems automatically inside a unified thread:

```bash
chmod +x start.sh
./start.sh

```


## Manual Setup

### 1. Initialize the Environment and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Rjchauhan18/CredIdentity.git
cd CredIdentity

# Sync your dependencies using the locked file format
uv sync

```

### 2. (Optional) Generate and Engineer the Features
`Note:`This command is used for generating mock data.

```bash
# Generate the synthetic data base matrix
uv run python data_pipeline/generate_data.py

# Process the raw records into the 12 final engineered feature csv variables
uv run python data_pipeline/engineer_features.py

```

### 3. Execution Commands

To test the applications locally, use the following execution configurations:

* **To run the FastAPI Backend Engine:**
```bash
uv run uvicorn backend.main:app --reload

```


* **To run the Streamlit Frontend Dashboard UI:**
```bash
uv run streamlit run dashboard/app.py

```


---

## ☁️ Hugging Face Spaces Cloud Deployment Architecture

This project is optimized for deployment as a single **Docker-based Space** on Hugging Face (utilizing the free basic tier: **16 GB RAM / 2 vCPU**).

The included `Dockerfile` and `start.sh` handle running the dual processes in parallel inside a secure container:

* **Internal Port 8000:** Handles the FastAPI prediction engine, loading weights from the public Hugging Face model repository via `huggingface_hub` snapshot downloads.
* **Public Port 7860:** Exposes the Streamlit interface to users and judges.

### 🔄 Automated CI/CD Deployment via GitHub Actions

We eliminate the need for manual file transfers by automating cloud syncing. Every time you push to the `main` branch of this GitHub repository, the file `.github/workflows/hf_sync.yml` triggers automatically to push updates to Hugging Face via standard secure git remotes.

**Required Repo Environments:**

* Add your secure `HF_TOKEN` (Read access) and `HF_MODEL_REPO` as secrets on Hugging Face space variables.
* Set `HF_TOKEN_WRITE` as an action repository secret inside GitHub to allow the runner automation to build your release flawlessly.
