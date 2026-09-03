"""Run the prepared replay sample and preserve an immutable evidence bundle."""

from __future__ import annotations

import json

import pandas as pd

from ml_nids.config import ARTIFACTS_DIR, REPLAY_DIR, RESULTS_DIR
from ml_nids.governance import sha256_file
from ml_nids.replay import (
    load_trusted_bundle,
    run_controlled_replay,
    validate_replay_frame,
)

BATCH_SIZE = 1_000
OUTPUT_DIR = RESULTS_DIR / "dashboard"
PREDICTIONS_PATH = OUTPUT_DIR / "replay_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "replay_run.json"


if __name__ == "__main__":
    if PREDICTIONS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError("Replay evidence already exists and will not be overwritten")

    input_path = REPLAY_DIR / "replay_input.csv"
    bundle = load_trusted_bundle(
        ARTIFACTS_DIR, ARTIFACTS_DIR / "model_manifest.json", "reduced"
    )
    inputs = validate_replay_frame(
        pd.read_csv(input_path, low_memory=False), bundle["features"]
    )
    predictions, performance = run_controlled_replay(bundle, inputs, BATCH_SIZE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    summary = {
        "batch_size": BATCH_SIZE,
        "input_sha256": sha256_file(input_path),
        "model_condition": bundle["condition"],
        "model_training_sha256": bundle["training_sha256"],
        "rows": len(predictions),
        "prediction_counts": dict(
            sorted(predictions["prediction"].value_counts().to_dict().items())
        ),
        "alert_count": int(predictions["alert"].sum()),
        "elapsed_seconds": performance["elapsed_seconds"],
        "flows_per_second": performance["flows_per_second"],
        "predictions_sha256": sha256_file(PREDICTIONS_PATH),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

