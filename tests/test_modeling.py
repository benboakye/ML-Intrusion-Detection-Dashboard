import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from ml_nids.modeling import load_selected_features, train_and_freeze
from ml_nids.preparation import DataPreparationError


def _training_frame(rows_per_class: int = 30) -> pd.DataFrame:
    rows = []
    for class_number, label in enumerate(("BENIGN", "SYN", "UDP")):
        for index in range(rows_per_class):
            row = {"Label": label}
            for feature_number in range(16):
                row[f"feature_{feature_number:02d}"] = (
                    class_number * 20 + index * (feature_number + 1)
                ) % 97
            rows.append(row)
    return pd.DataFrame(rows)


def test_selected_features_require_exact_frozen_count(tmp_path: Path) -> None:
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(["feature_00"]), encoding="utf-8")

    with pytest.raises(DataPreparationError, match="15 unique"):
        load_selected_features(path, ["feature_00"])


def test_training_uses_same_configuration_and_writes_trusted_bundles(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "train.parquet"
    selected_path = tmp_path / "selected.json"
    artifacts = tmp_path / "artifacts"
    frame = _training_frame()
    frame.to_parquet(training_path, index=False)
    selected = [f"feature_{number:02d}" for number in range(15)]
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    parameters = {
        "n_estimators": 5,
        "max_depth": 4,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "max_samples": 0.7,
        "class_weight": "balanced_subsample",
        "n_jobs": 1,
        "random_state": 42,
    }

    result = train_and_freeze(
        training_path=training_path,
        selected_features_path=selected_path,
        artifacts_dir=artifacts,
        validation_path=tmp_path / "validation.json",
        freeze_path=tmp_path / "freeze.json",
        parameters=parameters,
    )

    assert result["validation"]["external_test_used"] is False
    assert result["freeze"]["external_test_evaluated"] is False
    assert set(result["model_manifest"]["models"]) == {"full", "reduced"}
    for condition, expected_count in (("full", 16), ("reduced", 15)):
        path = artifacts / f"{condition}_rf.joblib"
        bundle = joblib.load(path)
        assert path.is_file()
        assert len(bundle["features"]) == expected_count
        assert set(bundle["classes"]) == {"BENIGN", "SYN", "UDP"}
        assert len(result["model_manifest"]["models"][condition]["sha256"]) == 64
