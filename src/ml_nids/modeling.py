"""Fair full-versus-reduced Random Forest training and validation."""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_nids.analysis import load_training_data
from ml_nids.config import (
    LABEL_COLUMN,
    MAX_ATTACK_RECALL_LOSS,
    MAX_MACRO_F1_LOSS,
    RANDOM_FOREST_PARAMS,
    REDUCED_FEATURE_COUNT,
    SEED,
    TARGET_LABELS,
)
from ml_nids.governance import sha256_file
from ml_nids.preparation import DataPreparationError


def build_pipeline(
    parameters: Mapping[str, Any] = RANDOM_FOREST_PARAMS,
) -> Pipeline:
    """Build the shared preprocessing/model pipeline used by both conditions."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**dict(parameters))),
        ]
    )


def load_selected_features(path: Path, available: list[str]) -> tuple[str, ...]:
    """Load and validate the frozen reduced feature list."""

    try:
        selected = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPreparationError(f"Could not read selected features: {path}") from exc
    if not isinstance(selected, list) or not all(
        isinstance(feature, str) for feature in selected
    ):
        raise DataPreparationError("Selected features must be a JSON string array")
    if len(selected) != REDUCED_FEATURE_COUNT or len(set(selected)) != len(selected):
        raise DataPreparationError(
            f"Selected features must contain {REDUCED_FEATURE_COUNT} unique names"
        )
    missing = sorted(set(selected) - set(available))
    if missing:
        raise DataPreparationError(f"Selected features are unavailable: {missing}")
    return tuple(selected)


def _condition_metrics(y_true: pd.Series, prediction: Any) -> dict[str, Any]:
    report = classification_report(
        y_true,
        prediction,
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
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
        "confusion_matrix": confusion_matrix(
            y_true, prediction, labels=TARGET_LABELS
        ).tolist(),
    }


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dump_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def train_and_freeze(
    *,
    training_path: Path,
    selected_features_path: Path,
    artifacts_dir: Path,
    validation_path: Path,
    freeze_path: Path,
    parameters: Mapping[str, Any] = RANDOM_FOREST_PARAMS,
) -> dict[str, Any]:
    """Validate both conditions fairly, retrain on all training rows, and freeze."""

    frame = load_training_data(training_path)
    labels = frame[LABEL_COLUMN]
    features = frame.drop(columns=LABEL_COLUMN)
    full_features = tuple(features.columns)
    reduced_features = load_selected_features(
        selected_features_path, list(full_features)
    )

    development_rows, validation_rows = train_test_split(
        frame.index,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    validation: dict[str, Any] = {
        "seed": SEED,
        "development_rows": len(development_rows),
        "validation_rows": len(validation_rows),
        "external_test_used": False,
        "conditions": {},
    }
    model_manifest: dict[str, Any] = {
        "training_file": training_path.name,
        "training_sha256": sha256_file(training_path),
        "selected_features_sha256": sha256_file(selected_features_path),
        "random_forest_parameters": dict(parameters),
        "models": {},
    }

    for condition, condition_features in (
        ("full", full_features),
        ("reduced", reduced_features),
    ):
        validation_pipeline = build_pipeline(parameters)
        started = time.perf_counter()
        validation_pipeline.fit(
            features.loc[development_rows, condition_features],
            labels.loc[development_rows],
        )
        validation_prediction = validation_pipeline.predict(
            features.loc[validation_rows, condition_features]
        )
        condition_result = _condition_metrics(
            labels.loc[validation_rows], validation_prediction
        )
        condition_result["fit_and_predict_seconds"] = time.perf_counter() - started
        condition_result["feature_count"] = len(condition_features)
        validation["conditions"][condition] = condition_result

        final_pipeline = build_pipeline(parameters)
        started = time.perf_counter()
        final_pipeline.fit(features.loc[:, condition_features], labels)
        fit_seconds = time.perf_counter() - started
        bundle = {
            "pipeline": final_pipeline,
            "features": list(condition_features),
            "classes": list(final_pipeline.classes_),
            "condition": condition,
            "training_sha256": model_manifest["training_sha256"],
        }
        model_path = artifacts_dir / f"{condition}_rf.joblib"
        _dump_joblib(bundle, model_path)
        model_manifest["models"][condition] = {
            "file": model_path.name,
            "feature_count": len(condition_features),
            "fit_seconds": fit_seconds,
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
        }

    _write_json(validation, validation_path)
    _write_json(model_manifest, artifacts_dir / "model_manifest.json")
    freeze = {
        "freeze_date": date.today().isoformat(),
        "training_sha256": model_manifest["training_sha256"],
        "class_counts": dict(sorted(labels.value_counts().to_dict().items())),
        "full_feature_count": len(full_features),
        "reduced_feature_count": len(reduced_features),
        "reduced_features": list(reduced_features),
        "random_forest_parameters": dict(parameters),
        "validation_metrics": {
            condition: {
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
                "recall": {
                    label: values["per_class"][label]["recall"]
                    for label in TARGET_LABELS
                },
            }
            for condition, values in validation["conditions"].items()
        },
        "acceptance_margins": {
            "maximum_macro_f1_loss": MAX_MACRO_F1_LOSS,
            "maximum_syn_or_udp_recall_loss": MAX_ATTACK_RECALL_LOSS,
        },
        "model_hashes": {
            condition: values["sha256"]
            for condition, values in model_manifest["models"].items()
        },
        "external_test_evaluated": False,
    }
    _write_json(freeze, freeze_path)
    return {
        "validation": validation,
        "model_manifest": model_manifest,
        "freeze": freeze,
    }
