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

* **Internal Port 8000:** Handles the FastAPI prediction engine, loading weights directly from your secure private repository via `huggingface_hub` snapshot downloads.
* **Public Port 7860:** Exposes the Streamlit interface to users and judges.

### 🔄 Automated CI/CD Deployment via GitHub Actions

We eliminate the need for manual file transfers by automating cloud syncing. Every time you push to the `main` branch of this GitHub repository, the file `.github/workflows/hf_sync.yml` triggers automatically to push updates to Hugging Face via standard secure git remotes.

**Required Repo Environments:**

* Add your secure `HF_TOKEN` (Read access) and `HF_MODEL_REPO` as secrets on Hugging Face space variables.
* Set `HF_TOKEN_WRITE` as an action repository secret inside GitHub to allow the runner automation to build your release flawlessly.
