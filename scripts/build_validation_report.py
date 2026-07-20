"""Offline validation + business-outcome backtest for the slim credit model.

Scores the model over a held-out slice of the labelled dataset ONCE and writes a
compact JSON artifact the app serves read-only. Nothing here runs at app-boot; the
dashboards just render these precomputed numbers.

Outputs `model/validation_report.json` with:
  - discrimination metrics: ROC-AUC, Gini, KS statistic
  - a calibration table (probability deciles: predicted vs observed default rate)
  - a business backtest: approval-rate vs bad-rate as the score cutoff sweeps 300-900

Usage:
    python -m scripts.build_validation_report
    python -m scripts.build_validation_report --data data/processed/msme_final_engineered_features.csv --n 20000

The held-out slice uses a fixed seed so the report is reproducible. If a canonical
train/test split is later exported from the notebook, point --data at the test file
and pass --no-split.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Reuse the exact serving constants so the report can't drift from production scoring.
from backend.credit_engine import (
    NON_FEATURE_COLUMNS,
    _score_from_prob,
    _tier_for,
    init_model,
)
import backend.credit_engine as engine

RANDOM_SEED = 42
DEFAULT_DATA = "data/processed/msme_final_engineered_features.csv"
OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/validation_report.json"))


def _ks_statistic(y_true: np.ndarray, prob_default: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between the cumulative distributions of
    predicted default probability for the bad (y=1) and good (y=0) populations."""
    order = np.argsort(prob_default)
    y = y_true[order]
    cum_bad = np.cumsum(y) / max(1, y.sum())
    cum_good = np.cumsum(1 - y) / max(1, (1 - y).sum())
    return float(np.max(np.abs(cum_bad - cum_good)))


def _roc_auc(y_true: np.ndarray, prob_default: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), no sklearn dependency required."""
    order = np.argsort(prob_default)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(prob_default) + 1)
    # Average ranks for ties so AUC is exact on discrete probabilities.
    df = pd.DataFrame({"p": prob_default, "r": ranks})
    df["r"] = df.groupby("p")["r"].transform("mean")
    ranks = df["r"].to_numpy()
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _calibration_table(y_true: np.ndarray, prob_default: np.ndarray, bins: int = 10) -> list:
    """Decile calibration: mean predicted default prob vs observed default rate."""
    df = pd.DataFrame({"y": y_true, "p": prob_default})
    # qcut with duplicate-edge tolerance; fall back to fewer bins if probs are degenerate.
    try:
        df["bucket"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    except ValueError:
        df["bucket"] = pd.cut(df["p"], bins=bins)
    rows = []
    for i, (interval, g) in enumerate(df.groupby("bucket", observed=True)):
        rows.append({
            "decile": i + 1,
            "predicted_default_rate": round(float(g["p"].mean()), 4),
            "observed_default_rate": round(float(g["y"].mean()), 4),
            "count": int(len(g)),
        })
    return rows


def _backtest(y_true: np.ndarray, scores: np.ndarray) -> list:
    """Business-outcome backtest. For each score cutoff, 'approve' applicants at or
    above it and report approval rate, bad-rate among approved, and defaults avoided."""
    total = len(y_true)
    total_bad = int(y_true.sum())
    rows = []
    for cutoff in range(300, 901, 25):
        approved = scores >= cutoff
        n_appr = int(approved.sum())
        bad_appr = int(y_true[approved].sum())
        rows.append({
            "cutoff": cutoff,
            "approval_rate": round(n_appr / total, 4),
            "approved_count": n_appr,
            "bad_rate_among_approved": round(bad_appr / n_appr, 4) if n_appr else 0.0,
            "defaults_avoided": total_bad - bad_appr,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--n", type=int, default=20000, help="held-out sample size")
    parser.add_argument("--no-split", action="store_true", help="treat --data as an already held-out set")
    args = parser.parse_args()

    print("🧠 Loading model (this also populates serving constants)...")
    init_model()
    predictor = engine.predictor
    if predictor is None:
        raise SystemExit("Model failed to load; cannot build validation report.")

    print(f"📊 Reading {args.data} ...")
    # Some company names carry cp1252 punctuation (e.g. curly apostrophes), so fall
    # back to latin-1 if the file isn't clean UTF-8. Only the label/feature columns
    # are used downstream, so a lossy decode of name text is harmless here.
    try:
        df = pd.read_csv(args.data)
    except UnicodeDecodeError:
        df = pd.read_csv(args.data, encoding="latin-1")
    if "is_defaulter" not in df.columns:
        raise SystemExit("Dataset has no is_defaulter label; cannot validate.")

    # Held-out slice: stratified-ish sample with a fixed seed for reproducibility.
    if not args.no_split and len(df) > args.n:
        df = df.groupby("is_defaulter", group_keys=False).apply(
            lambda g: g.sample(min(len(g), int(args.n * len(g) / len(df)) + 1), random_state=RANDOM_SEED)
        )
    print(f"   scoring {len(df):,} held-out rows...")

    X = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    y = df["is_defaulter"].to_numpy().astype(int)

    proba = predictor.predict_proba(X)
    default_label = 1 if 1 in proba.columns else predictor.class_labels[-1]
    healthy_label = 0 if 0 in proba.columns else predictor.class_labels[0]
    prob_default = proba[default_label].to_numpy()
    prob_healthy = proba[healthy_label].to_numpy()
    scores = np.array([_score_from_prob(p) for p in prob_healthy])

    auc = _roc_auc(y, prob_default)
    gini = 2 * auc - 1
    ks = _ks_statistic(y, prob_default)

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_repo": os.getenv("HF_MODEL_REPO", "unknown"),
        "sample_size": int(len(df)),
        "default_rate": round(float(y.mean()), 4),
        "discrimination": {
            "roc_auc": round(auc, 4),
            "gini": round(gini, 4),
            "ks_statistic": round(ks, 4),
        },
        "calibration": _calibration_table(y, prob_default),
        "backtest": _backtest(y, scores),
        "score_distribution": {
            "min": int(scores.min()),
            "p25": int(np.percentile(scores, 25)),
            "median": int(np.percentile(scores, 50)),
            "p75": int(np.percentile(scores, 75)),
            "max": int(scores.max()),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Wrote {OUT_PATH}")
    print(f"   AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}  default_rate={y.mean():.4f}")


if __name__ == "__main__":
    main()
