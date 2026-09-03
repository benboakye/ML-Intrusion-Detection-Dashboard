"""Run the frozen models once on the independent external-test day."""

from ml_nids.config import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR
from ml_nids.evaluation import evaluate_frozen_models


if __name__ == "__main__":
    record = evaluate_frozen_models(
        external_test_path=PROCESSED_DIR / "external_test.parquet",
        preparation_summary_path=ARTIFACTS_DIR / "data_preparation_summary.json",
        model_manifest_path=ARTIFACTS_DIR / "model_manifest.json",
        freeze_path=RESULTS_DIR / "validation" / "freeze.json",
        artifacts_dir=ARTIFACTS_DIR,
        output_dir=RESULTS_DIR / "final_external_test",
    )
    for condition, result in record["conditions"].items():
        print(
            f"{condition}: accuracy={result['accuracy']:.4f}; "
            f"macro_f1={result['macro_f1']:.4f}; "
            f"median={result['benchmark']['median_seconds']:.4f}s"
        )
    print(f"acceptance passed: {record['acceptance']['passed']}")
