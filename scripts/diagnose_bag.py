"""
Focused diagnostic: WHY does keeping only LightGBM_r96_BAG_L1 give NoneType.predict?

We do NOT trim/copy anything here. We load the FULL model read-only and interrogate
the bagged model's internal structure directly, so we can see the true cause instead
of guessing. Fast: one load, no copytree, no delete.
"""
import warnings
warnings.filterwarnings("ignore")
from autogluon.tabular import TabularPredictor

SRC = "model/local_model_weights"
TARGET = "LightGBM_r96_BAG_L1"

p = TabularPredictor.load(SRC, require_py_version_match=False)
trainer = p._trainer

print("=" * 70)
print(f"Inspecting {TARGET}")
print("=" * 70)

# 1) What does the trainer think this model depends on?
try:
    print("minimum_model_set:", trainer.get_minimum_model_set(TARGET))
except Exception as e:
    print("get_minimum_model_set error:", repr(e))

# 2) Load the model OBJECT and look inside the bag.
m = trainer.load_model(TARGET)
print("\nloaded model type:", type(m).__name__)
print("has .models attr:", hasattr(m, "models"))
print("model_base:", getattr(m, "model_base", "N/A"))
# BaggedEnsembleModel stores child fold models in .models (names or objects)
children = getattr(m, "models", None)
print("m.models:", children)
print("len(m.models):", len(children) if children is not None else "N/A")

# 3) Are the child fold binaries actually on disk?
import os
mdir = os.path.join(SRC, "models", TARGET)
print(f"\nfiles under {mdir}:")
for root, _, files in os.walk(mdir):
    for f in files:
        fp = os.path.join(root, f)
        print(f"   {os.path.relpath(fp, mdir)}  ({os.path.getsize(fp)} bytes)")

# 4) Try the actual failing call in isolation, low_memory vs not.
import pandas as pd
df = pd.read_csv("data/processed/msme_demo_features.csv").drop(
    columns=["is_defaulter", "risk_band", "bounce_mandate_failure_rate",
             "msme_id", "company_name"], errors="ignore").head(1)

try:
    m2 = trainer.load_model(TARGET)
    print(f"\nbag has load_child_models(): {hasattr(m2, 'load_child_models')}")
    pr = p.predict_proba(df, model=TARGET)
    print(f"[predict via predictor, model={TARGET}] OK -> {pr.values.tolist()}")
except Exception as e:
    print(f"[predict model={TARGET}] FAILED: {repr(e)}")

# 5) Inspect the persisted child fold directly
print("\n--- child fold inspection ---")
try:
    m3 = trainer.load_model(TARGET)
    if getattr(m3, "models", None):
        for cname in m3.models:
            print("child entry:", cname, type(cname).__name__)
            # If the bag stores child NAMES (not objects), load each and report .model
            if isinstance(cname, str):
                try:
                    child = trainer.load_model(cname)
                    print("   loaded child:", type(child).__name__,
                          "| inner .model is None?:", getattr(child, "model", "NO_ATTR") is None)
                except Exception as ce:
                    print("   child load FAILED:", repr(ce))
    else:
        print("bag has no .models list populated (children not loaded into parent)")
    # Direct low-level predict on the bag object itself, bypassing the predictor wrapper
    print("\n--- direct bag.predict_proba attempt ---")
    try:
        Xt = p.transform_features(df)
        out = m3.predict_proba(Xt)
        print("bag.predict_proba OK ->", out.tolist() if hasattr(out, "tolist") else out)
    except Exception as be:
        print("bag.predict_proba FAILED:", repr(be))
except Exception as e:
    print("child fold inspection error:", repr(e))
