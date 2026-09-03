"""Build and run safe, reproducible controlled-replay inputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_nids.config import (
    DASHBOARD_MAX_ROWS,
    LABEL_COLUMN,
    REPLAY_ROWS_PER_CLASS,
    SEED,
    TARGET_LABELS,
)
from ml_nids.governance import sha256_file
from ml_nids.preparation import DataPreparationError


def make_replay_sample(
    external_test_path: Path,
    *,
    input_path: Path,
    truth_path: Path,
    manifest_path: Path,
    rows_per_class: int = REPLAY_ROWS_PER_CLASS,
) -> dict[str, Any]:
    """Create a labelled audit pair while keeping truth out of dashboard input."""

    if rows_per_class <= 0:
        raise ValueError("rows_per_class must be positive")
    frame = pd.read_parquet(external_test_path)
    if LABEL_COLUMN not in frame or set(frame[LABEL_COLUMN]) != set(TARGET_LABELS):
        raise DataPreparationError("External-test data does not contain target labels")
    parts = []
    for label in TARGET_LABELS:
        group = frame.loc[frame[LABEL_COLUMN].eq(label)]
        parts.append(group.sample(n=min(rows_per_class, len(group)), random_state=SEED))
    sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)
    sample = sample.reset_index(drop=True)
    input_frame = sample.drop(columns=LABEL_COLUMN)
    truth_frame = sample[[LABEL_COLUMN]].rename(columns={LABEL_COLUMN: "true_label"})
    truth_frame.index.name = "replay_row"

    for path in (input_path, truth_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    input_frame.to_csv(input_path, index=False)
    truth_frame.to_csv(truth_path, index=True)
    manifest = {
        "seed": SEED,
        "rows_per_class_maximum": rows_per_class,
        "rows": len(sample),
        "class_counts": dict(sorted(sample[LABEL_COLUMN].value_counts().to_dict().items())),
        "external_test_sha256": sha256_file(external_test_path),
        "input": {
            "file": input_path.name,
            "sha256": sha256_file(input_path),
            "size_bytes": input_path.stat().st_size,
            "contains_label": False,
        },
        "truth": {
            "file": truth_path.name,
            "sha256": sha256_file(truth_path),
            "size_bytes": truth_path.stat().st_size,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_trusted_bundle(
    artifacts_dir: Path, model_manifest_path: Path, condition: str = "reduced"
) -> dict[str, Any]:
    """Hash-check and load a locally produced model bundle."""

    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("models", {}).get(condition)
    if not isinstance(record, dict) or not isinstance(record.get("file"), str):
        raise DataPreparationError(f"Model manifest is missing {condition}")
    path = (artifacts_dir / record["file"]).resolve()
    if not path.is_relative_to(artifacts_dir.resolve()):
        raise DataPreparationError("Model path leaves the artifact directory")
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise DataPreparationError("Trusted model hash verification failed")
    bundle = joblib.load(path)  # trusted local artifact after hash verification
    if bundle.get("condition") != condition:
        raise DataPreparationError("Model bundle condition does not match manifest")
    if set(bundle.get("classes", [])) != set(TARGET_LABELS):
        raise DataPreparationError("Model bundle classes do not match targets")
    return bundle


def validate_replay_frame(
    frame: pd.DataFrame,
    required_features: list[str],
    *,
    maximum_rows: int = DASHBOARD_MAX_ROWS,
) -> pd.DataFrame:
    """Validate a prepared replay upload and return numeric model inputs."""

    if frame.empty:
        raise DataPreparationError("Replay CSV contains no rows")
    if len(frame) > maximum_rows:
        raise DataPreparationError(
            f"Replay CSV has {len(frame):,} rows; maximum is {maximum_rows:,}"
        )
    normalized = [str(column).strip() for column in frame.columns]
    if len(normalized) != len(set(normalized)):
        raise DataPreparationError("Replay columns are duplicated after trimming")
    frame = frame.copy()
    frame.columns = normalized
    missing = sorted(set(required_features) - set(frame.columns))
    if missing:
        raise DataPreparationError(f"Replay CSV is missing: {', '.join(missing)}")
    inputs = frame.loc[:, required_features].apply(pd.to_numeric, errors="coerce")
    return inputs.replace([np.inf, -np.inf], np.nan)


def run_controlled_replay(
    bundle: dict[str, Any], inputs: pd.DataFrame, batch_size: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Predict prepared rows in timed batches without generating network traffic."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    features = bundle["features"]
    missing = sorted(set(features) - set(inputs.columns))
    if missing:
        raise DataPreparationError(f"Validated replay data is missing: {missing}")
    records = []
    overall_started = time.perf_counter()
    for start in range(0, len(inputs), batch_size):
        batch = inputs.iloc[start : start + batch_size]
        started = time.perf_counter()
        prediction = bundle["pipeline"].predict(batch.loc[:, features])
        latency = time.perf_counter() - started
        records.append(
            pd.DataFrame(
                {
                    "replay_row": batch.index,
                    "prediction": prediction,
                    "batch_latency_seconds": latency,
                }
            )
        )
    elapsed = time.perf_counter() - overall_started
    result = pd.concat(records, ignore_index=True)
    result["alert"] = result["prediction"].ne("BENIGN")
    return result, {
        "elapsed_seconds": elapsed,
        "flows_per_second": len(result) / elapsed,
    }
