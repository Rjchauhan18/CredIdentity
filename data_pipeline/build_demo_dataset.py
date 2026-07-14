"""Build a small, committable demo dataset for the dashboard/API.

The main training data (250k rows) is large and gitignored, so it never reaches
Hugging Face Spaces. This script generates a handful of MSME profiles using the
SAME generator and distribution as the training set, but with a seed offset
(`DEMO_START_INDEX`) that guarantees these rows were NOT seen during training.

Output: data/processed/msme_demo_features.csv  (small enough to commit)

Usage:
    uv run python data_pipeline/build_demo_dataset.py
"""
import os
import tempfile

from data_pipeline.generate_data import generate_msme_raw_data
from data_pipeline.engineer_features import calculate_12_features

# Far outside the training index range (0..249_999) so seeds never overlap ->
# genuinely held-out, same-distribution rows.
DEMO_START_INDEX = 1_000_000
DEMO_ROWS = 50
OUTPUT_CSV = "data/processed/msme_demo_features.csv"


def main():
    os.makedirs("data/processed", exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        raw_path = tmp.name
    try:
        generate_msme_raw_data(
            num_records=DEMO_ROWS,
            output_path=raw_path,
            workers=1,
            start_index=DEMO_START_INDEX,
        )
        calculate_12_features(raw_path, OUTPUT_CSV)
        print(f"✅ Demo dataset ({DEMO_ROWS} held-out rows) written to {OUTPUT_CSV}")
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


if __name__ == "__main__":
    main()
