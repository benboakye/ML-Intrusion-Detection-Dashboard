"""Generate training-only exploratory data evidence."""

from ml_nids.analysis import load_training_data, write_exploration
from ml_nids.config import PROCESSED_DIR, RESULTS_DIR


if __name__ == "__main__":
    source = PROCESSED_DIR / "train.parquet"
    outputs = write_exploration(load_training_data(source), RESULTS_DIR / "eda")
    print("training-only EDA outputs:")
    for output in outputs:
        print(f"- {output}")
