"""
Slim the AutoGluon model for real-time serving (Problem Statement #3: "near real-time").

WHAT THIS DOES
--------------
The model is trained with AutoGluon's `best_quality` preset, which produces a large
multi-layer stacking ensemble (400+ files). That is great for peak accuracy but slow to
load (~6 min cold start) and memory-heavy at inference. This script distills it to a
deployment model that keeps only the artifacts the *best* model actually needs, then
PROVES the credit scores are unchanged on the held-out demo cohort before you re-upload.

It is NON-DESTRUCTIVE: the source model is never modified. A new slim copy is written to
the target directory. Re-uploading the slim copy to the HF model repo is a manual step
you stay in control of.

USAGE (inside the container, which has AutoGluon + the model):
    python scripts/slim_model.py \
        --source model/local_model_weights \
        --target model/slim_model_weights \
        --demo   data/processed/msme_demo_features.csv
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Must match backend/credit_engine.py — columns stripped before inference.
NON_FEATURE_COLUMNS = ["is_defaulter", "risk_band", "bounce_mandate_failure_rate"]
# Non-model identifier columns present in the demo CSV.
ID_COLUMNS = ["msme_id", "company_name"]


def _dir_stats(path: str):
    """Return (file_count, total_megabytes) for a model directory."""
    n, total = 0, 0
    for root, _, files in os.walk(path):
        for f in files:
            n += 1
            total += os.path.getsize(os.path.join(root, f))
    return n, total / (1024 * 1024)


def _to_credit_score(prob_healthy: float) -> int:
    """Same 300-900 mapping the backend uses, so the delta is in real score points."""
    return max(300, min(900, int(300 + (prob_healthy * 600))))


def main():
    from autogluon.tabular import TabularPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="model/local_model_weights")
    ap.add_argument("--target", default="model/slim_model_weights")
    ap.add_argument("--demo", default="data/processed/msme_demo_features.csv")
    # keep-mode:
    #   "best"   -> keep the full WeightedEnsemble AutoGluon serves (0 drift, still multi-GB)
    #   a model name (e.g. "WeightedEnsemble_L2" or "XGBoost_BAG_L2") -> keep just that,
    #      giving a tiny model that cold-starts in seconds (may shift scores slightly).
    ap.add_argument("--keep", default="best",
                    help="'best', 'single', 'ensemble_l2', 'ensemble_topn', or a model name")
    ap.add_argument("--topn", type=int, default=12,
                    help="for --keep ensemble_topn: how many top L1 base models to blend")
    args = ap.parse_args()

    import shutil

    print("=" * 70)
    print("STEP 1 — Load the current (full) model and show the leaderboard")
    print("=" * 70)
    full = TabularPredictor.load(args.source, require_py_version_match=False)
    best = full.model_best
    lb = full.leaderboard(silent=True)[["model", "score_val"]]
    print(f"Best model AutoGluon serves by default: {best}\n")
    print(lb.to_string(index=False))

    # Drop rows with no validation score (the *_FULL refit variants), then find the
    # best non-ensemble single model so we can quote the accuracy gap honestly.
    scored = lb.dropna(subset=["score_val"]).copy()
    scored["score_val"] = scored["score_val"].astype(float)
    ensemble_val = scored.loc[scored["model"].str.contains("WeightedEnsemble"), "score_val"].max()
    single = scored[~scored["model"].str.contains("WeightedEnsemble")]
    top_single = single.loc[single["score_val"].idxmax()]
    print(
        f"\nBest ensemble val AUC: {ensemble_val:.5f}"
        f"\nBest single-model val AUC: {top_single['model']} = {top_single['score_val']:.5f}"
        f"\nAccuracy given up by dropping the full stack: {ensemble_val - top_single['score_val']:+.5f} AUC"
    )

    src_files, src_mb = _dir_stats(args.source)
    print(f"\nSource model footprint: {src_files} files, {src_mb:.1f} MB")

    # Decide which model to keep. "single" auto-picks the best L1 model, which has NO
    # base-model dependencies -> the resulting artifact is tiny and cold-starts in seconds.
    if args.keep == "single":
        # Restrict to TREE-based L1 models (CatBoost/XGBoost/LightGBM). Neural nets
        # (FastAI/Torch) don't survive save_space() for serving — FastAI drops its
        # dataloader and crashes on predict. L1 => no stacking dependencies => tiny.
        tree_pat = "CatBoost|XGBoost|LightGBM|RandomForest|ExtraTrees"
        l1_tree = single[single["model"].str.contains("_L1") &
                         single["model"].str.contains(tree_pat) &
                         ~single["model"].str.contains("_FULL")]
        keep_model = l1_tree.loc[l1_tree["score_val"].idxmax(), "model"]
        print(f"\n--keep single -> auto-selected best L1 TREE model: {keep_model} "
              f"(val {l1_tree['score_val'].max():.5f})")
    elif args.keep == "best":
        keep_model = best
    elif args.keep == "ensemble_l2":
        # AutoGluon's trained WeightedEnsemble_L2 blends ~8 models (vs 34 for the L3 stack the
        # app serves). We serve its _FULL variant in Step 2 (the bagged L2 can't serve trimmed
        # — its base models recurse into missing fold children). Blend keeps fold-averaging
        # smoothing, so scores track production far closer than a lone single model.
        keep_model = "WeightedEnsemble_L2"
    elif args.keep == "ensemble_l3":
        # Serve WeightedEnsemble_L3_FULL — the EXACT ensemble production's model_best points to,
        # but the _FULL (non-bagged) variant. Blends all ~34 models, so drift should be ~0. The
        # _FULL variant drops the fold copies, so it's far smaller than the full 10 GB bagged
        # stack while serving without the fold-child crash.
        keep_model = "WeightedEnsemble_L3"
    elif args.keep == "ensemble_topn":
        # Build a FRESH weighted blend of the top-N L1 base models by val_score, then serve its
        # _FULL variant. More models than L2 (~12) -> closer to the 34-model production stack,
        # still tiny. The actual build happens after the copy loads (needs `slim`); here we just
        # pick the base-model list from the leaderboard we already have.
        base_pool = single[single["model"].str.contains("_L1") &
                           ~single["model"].str.contains("_FULL") &
                           ~single["model"].str.contains("WeightedEnsemble")]
        topn_base = (base_pool.sort_values("score_val", ascending=False)
                     .head(args.topn)["model"].tolist())
        print(f"\n--keep ensemble_topn -> will blend top {len(topn_base)} L1 base models:")
        for m in topn_base:
            print(f"    {m}")
        keep_model = None  # resolved after the blend is built in Step 2
    else:
        keep_model = args.keep
    print(f"Keeping model: {keep_model}")

    print("\n" + "=" * 70)
    print("STEP 2 — Build a slim deployment model")
    print("=" * 70)
    if os.path.exists(args.target):
        shutil.rmtree(args.target)
    # Copy the model, then trim the COPY in place. We operate on the already-loaded
    # predictor object (no reload) so the Python 3.12->3.10 version assertion — which
    # clone_for_deployment trips on internally — never fires.
    shutil.copytree(args.source, args.target)
    slim = TabularPredictor.load(args.target, require_py_version_match=False)
    # ROOT CAUSE (proven by scripts/diagnose_bag.py): a bagged model like
    # LightGBM_r96_BAG_L1 delegates predict to per-fold children S1F1/S1F2/S1F3 that are
    # persisted INSIDE the bag's own folder, NOT as standalone trainer entries. Ripped out
    # alone, the bag asks the trainer to load "S1F1" -> "Model does not exist" -> child is
    # None -> NoneType.predict. So a bagged model cannot serve in isolation, no matter what
    # we keep. Fix: use the NON-BAGGED *_FULL refit variant, which is a single booster
    # trained on all data with no fold children. AutoGluon builds these during training;
    # if missing we create it with refit_full().
    trainer = slim._trainer
    if args.keep == "ensemble_topn":
        # Build a fresh weighted ensemble from the chosen base models, with its _FULL refit in
        # one call (refit_full=True). fit_weighted_ensemble returns the new model name(s).
        print(f"\nBuilding weighted blend of {len(topn_base)} models (with _FULL refit)...")
        built = slim.fit_weighted_ensemble(base_models=topn_base, name_suffix="TopN",
                                           refit_full=True)
        print(f"fit_weighted_ensemble created: {built}")
        # AutoGluon names it "WeightedEnsemble_L2TopN" (+ "_FULL") — it inserts the stack level
        # before our suffix, so match on the "TopN" tag, not a fixed prefix. Prefer the bagged
        # base name (the _FULL is appended by the serving block below).
        topn_models = [m for m in built if "TopN" in m]
        bagged = [m for m in topn_models if not m.endswith("_FULL")]
        keep_model = (bagged or topn_models)[0]
        print(f"Blend base model: {keep_model}")

    if args.keep in ("ensemble_l2", "ensemble_l3", "ensemble_topn"):
        # Serve the _FULL ensemble variant: a weighted blend of NON-BAGGED _FULL base
        # models. Proven fact (single-model run): _FULL models serve without the fold-child
        # NoneType crash. The BAGGED WeightedEnsemble_L2 does NOT work — its children are the
        # per-fold boosters that aren't standalone-loadable (confirmed by traceback: bagged
        # ensemble -> child LightGBM -> self.model is None). So we keep the _FULL ensemble,
        # which keeps the multi-model smoothing while staying servable.
        all_models = set(slim.model_names())
        full_model = keep_model + "_FULL"
        if full_model not in all_models:
            print(f"{full_model} not found; calling refit_full({keep_model})...")
            refit_map = slim.refit_full(model=keep_model)
            full_model = refit_map.get(keep_model, full_model)
        print(f"Serving _FULL ENSEMBLE: {full_model}")
    else:
        # ROOT CAUSE (proven by scripts/diagnose_bag.py): a bagged model like
        # LightGBM_r96_BAG_L1 delegates predict to per-fold children S1F1/S1F2/S1F3 that are
        # persisted INSIDE the bag's own folder, NOT as standalone trainer entries. Ripped out
        # alone, the bag asks the trainer to load "S1F1" -> "Model does not exist" -> child is
        # None -> NoneType.predict. So a single bagged model cannot serve in isolation. Fix:
        # use the NON-BAGGED *_FULL refit variant (single booster, no fold children).
        all_models = set(slim.model_names())
        full_model = keep_model + "_FULL"
        if full_model not in all_models:
            print(f"{full_model} not found; calling refit_full({keep_model})...")
            refit_map = slim.refit_full(model=keep_model)
            full_model = refit_map.get(keep_model, full_model)
        print(f"Serving NON-BAGGED model: {full_model}")

    keep_set = slim._trainer.get_minimum_model_set(full_model)
    print(f"Keep set ({len(keep_set)}): {keep_set}")
    # BUGFIX (proven by traceback): delete_models() ends by recomputing best via
    # get_model_best() -> max(perfs, key=lambda i: (i[1], -i[2])). Every *_FULL model has
    # val_score/predict_time = None, so -None throws TypeError. For EACH _FULL model in the
    # keep-set (1 for single, ~8 for the ensemble), copy its bagged parent's scores across.
    # Parent name = strip the trailing "_FULL". The ensemble node itself has a real score.
    all_known = set(trainer.get_model_names())
    for m in keep_set:
        if not m.endswith("_FULL"):
            continue
        parent = m[: -len("_FULL")]
        if parent not in all_known:
            continue
        for attr in ("val_score", "predict_time", "predict_1_time", "fit_time"):
            parent_val = trainer.get_model_attribute(parent, attr, default=None)
            cur_val = trainer.get_model_attribute(m, attr, default=None)
            if cur_val is None and parent_val is not None:
                trainer.set_model_attribute(m, attr, parent_val)
                print(f"  copied {attr}={parent_val} from {parent} -> {m}")
    try:
        trainer.model_best = full_model
        trainer.save()
    except Exception:
        pass
    slim.delete_models(models_to_keep=keep_set, dry_run=False)
    # Re-assert the served-model pointer after delete.
    try:
        slim.set_model_best(full_model, save_trainer=True)
    except Exception:
        slim._trainer.model_best = full_model
        slim._trainer.save()
    keep_model = full_model  # everything downstream (scoring, verdict) uses the served model
    # NOTE: no save_space() — it strips artifacts the served model still needs.
    # Guard: prove the trimmed model can actually predict before we copy it out.
    _probe = slim.predict_proba(pd.read_csv(args.demo)
                                .drop(columns=NON_FEATURE_COLUMNS + ID_COLUMNS, errors="ignore")
                                .head(1))
    assert _probe is not None and len(_probe) == 1, "trimmed model failed test predict"
    slim_files, slim_mb = _dir_stats(args.target)
    print(f"Slim model footprint:   {slim_files} files, {slim_mb:.1f} MB")
    print(
        f"Reduction: {src_files - slim_files} fewer files "
        f"({100 * (1 - slim_files / src_files):.0f}%), "
        f"{src_mb - slim_mb:.1f} MB smaller ({100 * (1 - slim_mb / src_mb):.0f}%)"
    )

    print("\n" + "=" * 70)
    print("STEP 3 — Prove the score movement on the held-out demo cohort")
    print("=" * 70)
    df = pd.read_csv(args.demo)
    X = df.drop(columns=NON_FEATURE_COLUMNS + ID_COLUMNS, errors="ignore")

    def score_frame(predictor):
        # No explicit model arg -> mirrors exactly how backend/credit_engine.py scores.
        proba = predictor.predict_proba(X)
        healthy = 0 if 0 in proba.columns else predictor.class_labels[0]
        p = proba[healthy].to_numpy()
        return p, np.array([_to_credit_score(v) for v in p])

    p_full, cs_full = score_frame(full)
    p_slim, cs_slim = score_frame(slim)

    prob_delta = np.abs(p_full - p_slim)
    score_delta = np.abs(cs_full - cs_slim)

    print(f"Rows compared: {len(df)}")
    print(f"Probability delta (healthy):  max {prob_delta.max():.6f}, mean {prob_delta.mean():.6f}")
    print(f"Credit-score delta (300-900): max {int(score_delta.max())} pts, mean {score_delta.mean():.2f} pts")
    identical = int((score_delta == 0).sum())
    print(f"Companies with IDENTICAL credit score: {identical}/{len(df)}")

    # Per-company table, sorted worst drift first, so the reviewer sees exactly which
    # companies move and whether the max is one outlier or a broad pattern.
    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
    ids = df[id_col].astype(str).tolist() if id_col else [f"row_{i}" for i in range(len(df))]
    rows = sorted(
        zip(ids, cs_full, cs_slim, score_delta),
        key=lambda r: r[3], reverse=True,
    )
    print("\nPer-company credit scores (sorted by drift):")
    print(f"  {'company':<28}{'full':>7}{'slim':>7}{'delta':>7}")
    for cid, f_s, s_s, d in rows:
        flag = "" if d == 0 else ("  <-- band-risk" if d > 10 else "")
        print(f"  {cid[:28]:<28}{int(f_s):>7}{int(s_s):>7}{int(d):>7}{flag}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if score_delta.max() <= 2:
        print("✅ Scores are effectively unchanged (<=2 pts). Safe to re-upload the slim model.")
    elif score_delta.max() <= 10:
        print("⚠️  Small score drift (<=10 pts). Review the table above before re-uploading.")
    else:
        print("❌ Meaningful score drift. Consider keeping the WeightedEnsemble instead of a single model.")
    print(f"\nSlim model written to: {os.path.abspath(args.target)}")
    print("Next step (manual): re-upload that folder to the HF model repo, then the app")
    print("will cold-start in seconds instead of minutes. No backend code changes needed.")


if __name__ == "__main__":
    main()
