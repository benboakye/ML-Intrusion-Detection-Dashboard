"""Hash-gated, one-time evaluation of frozen models on the external test day."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
)

from ml_nids.config import (
    BENCHMARK_REPETITIONS,
    LABEL_COLUMN,
    MAX_ATTACK_RECALL_LOSS,
    MAX_MACRO_F1_LOSS,
    PROJECT_ROOT,
    TARGET_LABELS,
)
from ml_nids.governance import sha256_file
from ml_nids.preparation import DataPreparationError


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPreparationError(f"Could not read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DataPreparationError(f"Expected a JSON object: {path}")
    return value


def _verified_model_path(
    artifacts_dir: Path, model_record: dict[str, Any]
) -> Path:
    filename = model_record.get("file")
    if not isinstance(filename, str):
        raise DataPreparationError("Model manifest contains an invalid filename")
    path = (artifacts_dir / filename).resolve()
    if not path.is_relative_to(artifacts_dir.resolve()):
        raise DataPreparationError("Model manifest path leaves the artifact directory")
    if not path.is_file() or sha256_file(path) != model_record.get("sha256"):
        raise DataPreparationError(f"Model hash verification failed: {path}")
    return path


def _metrics(labels: pd.Series, prediction: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        labels,
        prediction,
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrices = multilabel_confusion_matrix(labels, prediction, labels=TARGET_LABELS)
    false_positive_rates = {}
    for index, label in enumerate(TARGET_LABELS):
        true_negative, false_positive, _, _ = matrices[index].ravel()
        false_positive_rates[label] = float(
            false_positive / max(1, false_positive + true_negative)
        )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score")
            }
            for label in TARGET_LABELS
        },
        "false_positive_rate": false_positive_rates,
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=TARGET_LABELS
        ).tolist(),
    }


def _benchmark_condition(
    bundle: dict[str, Any],
    test: pd.DataFrame,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = bundle.get("features")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise DataPreparationError("Model bundle contains an invalid feature list")
    missing = sorted(set(features) - set(test.columns))
    if missing:
        raise DataPreparationError(f"External test is missing model features: {missing}")
    if set(bundle.get("classes", [])) != set(TARGET_LABELS):
        raise DataPreparationError("Model bundle classes do not match target labels")

    inputs = test.loc[:, features]
    labels = test[LABEL_COLUMN]
    pipeline = bundle["pipeline"]
    pipeline.predict(inputs.iloc[: min(1_000, len(inputs))])
    process = psutil.Process()
    timings: list[dict[str, Any]] = []
    reference_prediction: np.ndarray | None = None
    for repetition in range(1, repetitions + 1):
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        prediction = pipeline.predict(inputs)
        elapsed = time.perf_counter() - started
        rss_after = process.memory_info().rss
        if reference_prediction is None:
            reference_prediction = prediction
        elif not np.array_equal(reference_prediction, prediction):
            raise DataPreparationError("Predictions changed across timing repetitions")
        timings.append(
            {
                "repetition": repetition,
                "seconds": elapsed,
                "flows_per_second": len(inputs) / elapsed,
                "rss_delta_bytes": max(0, rss_after - rss_before),
            }
        )
    assert reference_prediction is not None
    result = _metrics(labels, reference_prediction)
    seconds = np.array([record["seconds"] for record in timings])
    result["benchmark"] = {
        "repetitions": repetitions,
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.percentile(seconds, 95)),
        "median_flows_per_second": float(len(inputs) / np.median(seconds)),
        "maximum_rss_delta_bytes": max(
            record["rss_delta_bytes"] for record in timings
        ),
    }
    result["feature_count"] = len(features)
    return result, timings


def _acceptance(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = results["full"]
    reduced = results["reduced"]
    macro_loss = full["macro_f1"] - reduced["macro_f1"]
    recall_loss = {
        label: full["per_class"][label]["recall"]
        - reduced["per_class"][label]["recall"]
        for label in ("SYN", "UDP")
    }
    efficiency = {
        "median_latency": reduced["benchmark"]["median_seconds"]
        < full["benchmark"]["median_seconds"],
        "throughput": reduced["benchmark"]["median_flows_per_second"]
        > full["benchmark"]["median_flows_per_second"],
        "memory_delta": reduced["benchmark"]["maximum_rss_delta_bytes"]
        < full["benchmark"]["maximum_rss_delta_bytes"],
        "model_size": reduced["model_size_bytes"] < full["model_size_bytes"],
    }
    checks = {
        "macro_f1_loss_within_margin": macro_loss <= MAX_MACRO_F1_LOSS,
        "syn_recall_loss_within_margin": recall_loss["SYN"]
        <= MAX_ATTACK_RECALL_LOSS,
        "udp_recall_loss_within_margin": recall_loss["UDP"]
        <= MAX_ATTACK_RECALL_LOSS,
        "at_least_one_efficiency_measure_improved": any(efficiency.values()),
    }
    return {
        "macro_f1_loss": macro_loss,
        "attack_recall_loss": recall_loss,
        "efficiency_improvements": efficiency,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_outputs(
    output_dir: Path,
    record: dict[str, Any],
    timings: dict[str, list[dict[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        fields = [
            "condition",
            "feature_count",
            "accuracy",
            "macro_f1",
            "benign_recall",
            "syn_recall",
            "udp_recall",
            "median_seconds",
            "p95_seconds",
            "median_flows_per_second",
            "maximum_rss_delta_bytes",
            "model_size_bytes",
        ]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for condition in ("full", "reduced"):
            result = record["conditions"][condition]
            writer.writerow(
                {
                    "condition": condition,
                    "feature_count": result["feature_count"],
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "benign_recall": result["per_class"]["BENIGN"]["recall"],
                    "syn_recall": result["per_class"]["SYN"]["recall"],
                    "udp_recall": result["per_class"]["UDP"]["recall"],
                    "median_seconds": result["benchmark"]["median_seconds"],
                    "p95_seconds": result["benchmark"]["p95_seconds"],
                    "median_flows_per_second": result["benchmark"][
                        "median_flows_per_second"
                    ],
                    "maximum_rss_delta_bytes": result["benchmark"][
                        "maximum_rss_delta_bytes"
                    ],
                    "model_size_bytes": result["model_size_bytes"],
                }
            )
    with (output_dir / "timing_repetitions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "condition",
                "repetition",
                "seconds",
                "flows_per_second",
                "rss_delta_bytes",
            ],
        )
        writer.writeheader()
        for condition in ("full", "reduced"):
            for timing in timings[condition]:
                writer.writerow({"condition": condition, **timing})

    write_confusion_plots(record, output_dir)


def write_confusion_plots(record: dict[str, Any], output_dir: Path) -> None:
    """Render legible confusion matrices from an existing evaluation record."""

    os.environ.setdefault(
        "MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim" / "matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for condition in ("full", "reduced"):
        matrix = np.array(record["conditions"][condition]["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(6.4, 5.2))
        image = axis.imshow(matrix, cmap="Greys")
        axis.set(
            title=f"{condition.title()} Random Forest",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=range(len(TARGET_LABELS)),
            yticks=range(len(TARGET_LABELS)),
            xticklabels=TARGET_LABELS,
            yticklabels=TARGET_LABELS,
        )
        threshold = matrix.max() / 2
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.subplots_adjust(left=0.20, right=0.87, bottom=0.16, top=0.86)
        figure.savefig(output_dir / f"confusion_{condition}.png", dpi=200)
        plt.close(figure)


def evaluate_frozen_models(
    *,
    external_test_path: Path,
    preparation_summary_path: Path,
    model_manifest_path: Path,
    freeze_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    repetitions: int = BENCHMARK_REPETITIONS,
) -> dict[str, Any]:
    """Verify the freeze, evaluate both conditions, and atomically save evidence."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if output_dir.exists():
        raise DataPreparationError(
            f"Final external-test output already exists; refusing overwrite: {output_dir}"
        )
    preparation = _read_json(preparation_summary_path)
    manifest = _read_json(model_manifest_path)
    freeze = _read_json(freeze_path)
    expected_test_hash = preparation.get("outputs", {}).get("external_test", {}).get(
        "sha256"
    )
    actual_test_hash = sha256_file(external_test_path)
    if actual_test_hash != expected_test_hash:
        raise DataPreparationError("External-test data hash does not match preparation audit")
    if freeze.get("external_test_evaluated") is not False:
        raise DataPreparationError("Freeze record does not represent a pre-test state")

    test = pd.read_parquet(external_test_path)
    if set(test.get(LABEL_COLUMN, pd.Series(dtype=str))) != set(TARGET_LABELS):
        raise DataPreparationError("External-test labels do not match the target classes")
    condition_results: dict[str, dict[str, Any]] = {}
    timings: dict[str, list[dict[str, Any]]] = {}
    for condition in ("full", "reduced"):
        model_record = manifest.get("models", {}).get(condition)
        if not isinstance(model_record, dict):
            raise DataPreparationError(f"Model manifest is missing {condition}")
        model_path = _verified_model_path(artifacts_dir, model_record)
        bundle = joblib.load(model_path)  # trusted, locally produced and hash-verified
        result, timing = _benchmark_condition(bundle, test, repetitions)
        result["model_sha256"] = model_record["sha256"]
        result["model_size_bytes"] = model_path.stat().st_size
        condition_results[condition] = result
        timings[condition] = timing

    record: dict[str, Any] = {
        "external_test_file": external_test_path.name,
        "external_test_sha256": actual_test_hash,
        "external_test_rows": len(test),
        "freeze_sha256": sha256_file(freeze_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "labels": list(TARGET_LABELS),
        "conditions": condition_results,
        "acceptance": _acceptance(condition_results),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "total_memory_bytes": psutil.virtual_memory().total,
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_output = output_dir.with_name(f".{output_dir.name}.partial")
    if staging_output.exists():
        raise DataPreparationError(
            f"Staging output already exists; inspect it before retrying: {staging_output}"
        )
    _write_outputs(staging_output, record, timings)
    os.replace(staging_output, output_dir)
    return record
