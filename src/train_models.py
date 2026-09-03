"""Validate and freeze the full and reduced Random Forest conditions."""

from ml_nids.config import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR
from ml_nids.modeling import train_and_freeze


if __name__ == "__main__":
    result = train_and_freeze(
        training_path=PROCESSED_DIR / "train.parquet",
        selected_features_path=ARTIFACTS_DIR / "selected_features.json",
        artifacts_dir=ARTIFACTS_DIR,
        validation_path=RESULTS_DIR / "validation" / "metrics.json",
        freeze_path=RESULTS_DIR / "validation" / "freeze.json",
    )
    for condition, metrics in result["validation"]["conditions"].items():
        print(
            f"{condition}: accuracy={metrics['accuracy']:.4f}; "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    print("external test evaluated: False")
