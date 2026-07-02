# 🏦 IDBI CredIdentity: AI-Driven MSME Health Card & Underwriting Desk

An advanced, data-driven financial evaluation ecosystem designed for **IDBI Bank** to assess New-to-Credit (NTC) MSMEs. By leveraging digital footprints from India's digital public infrastructure (DPI)—including Account Aggregator networks, GST filing history, and EPFO contributions—this platform replaces traditional collateral-based lending with algorithmic risk profiling.

The system processes raw, chaotic transactional feeds into **12 Engineered Features**, grouping them into **4 Core Financial Pillars** to output an instantly queryable MSME Health Card and a verified credit risk assessment profile.

---

## ⚙️ Core Modules & Data Pipelines

### 1. Data Pipeline Factory (`data_pipeline/`)

Designed to bypass dependencies on live production core banking endpoints during implementation.

* **`generate_data.py`**: Uses specialized rule weights to model transactional anomalies over 180-day periods. Injects realistic industry hurdles including over-midnight payment velocities, systemic ACH mandate bounces, and employee headcount volatility.
* **`engineer_features.py`**: An automated processing engine that parses raw streams into the 12 primary indicators mapped to IDBI Bank credit compliance parameters:
* *Cash Flow Resiliency:* Average Monthly Balance, Inflow-to-Outflow Ratio, Bounce/Mandate Failure Rate, Balance Depletion Speed.
* *Compliance Risk:* GSTR1 vs GSTR3B Variance, Late GST Filing Frequency.
* *Commercial Revenue:* B2B vs B2C Revenue Ratio, Monthly Revenue Growth Rate.
* *Operational Stability:* Active Employee Trend, EPFO Wage Consistency Score, Daily UPI Velocity, Night Transaction Ratio.



### 2. Streamlit Analytical Interface (`dashboard/`)

The interface is targeted directly at internal credit operations and risk management teams:

* **MSME Health Card View:** Provides a fast, plain-English summary of a business's health score alongside a pre-computed grid matching them with active commercial loan offerings via OCEN 4.0.
* **Underwriting Desk:** Features structural safety verification including a **Challenger Model Disagreement flag** when model variances look suspect. It presents detailed feature impacts utilizing SHAP TreeExplainer metrics.
* **Executive Portfolio Overview:** Reads generated system footprints directly to supply macro portfolio health monitoring, balance runway distribution profiles, and dynamic warnings for systemic bottlenecks.

---

## 🚀 Installation & Local Execution

This ecosystem is optimized to run via the ultra-fast Python package manager `uv`. If you do not have `uv` installed, standard `pip` execution commands can be substituted.

### Step 1: Clone and Environment Setup

```bash

git clone https://github.com/Rjchauhan18/CredIdentity.git

# Navigate to the project root
cd CredIdentity


# Install foundational UI, modeling, and mock packages
uv sync

```

### Step 2: Trigger the Data Generation Engine

Generate the multi-layered simulation pools and transform them into clean feature matrices before spinning up your workspace:

```bash
# Generate the raw data records
python data_pipeline/generate_data.py

# Process raw data into structured features
python data_pipeline/engineer_features.py

```

### Step 3: Run the Dashboard

```bash
uv run streamlit run dashboard/app.py

```

---