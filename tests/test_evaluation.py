import json
from pathlib import Path

import pandas as pd
import pytest

from ml_nids.evaluation import evaluate_frozen_models
from ml_nids.governance import sha256_file
from ml_nids.modeling import train_and_freeze
from ml_nids.preparation import DataPreparationError


def _frame(rows_per_class: int = 25) -> pd.DataFrame:
    rows = []
    for class_number, label in enumerate(("BENIGN", "SYN", "UDP")):
        for index in range(rows_per_class):
            row = {"Label": label}
            for feature_number in range(16):
                row[f"feature_{feature_number:02d}"] = (
                    class_number * 30 + index * (feature_number + 1)
                ) % 101
            rows.append(row)
    return pd.DataFrame(rows)


def _frozen_fixture(tmp_path: Path) -> dict[str, Path]:
    training = tmp_path / "train.parquet"
    external = tmp_path / "external_test.parquet"
    selected = tmp_path / "selected.json"
    artifacts = tmp_path / "artifacts"
    validation = tmp_path / "validation.json"
    freeze = tmp_path / "freeze.json"
    _frame().to_parquet(training, index=False)
    _frame(12).to_parquet(external, index=False)
    selected.write_text(
        json.dumps([f"feature_{number:02d}" for number in range(15)]),
        encoding="utf-8",
    )
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
    train_and_freeze(
        training_path=training,
        selected_features_path=selected,
        artifacts_dir=artifacts,
        validation_path=validation,
        freeze_path=freeze,
        parameters=parameters,
    )
    preparation = tmp_path / "preparation.json"
    preparation.write_text(
        json.dumps({"outputs": {"external_test": {"sha256": sha256_file(external)}}}),
        encoding="utf-8",
    )
    return {
        "external_test_path": external,
        "preparation_summary_path": preparation,
        "model_manifest_path": artifacts / "model_manifest.json",
        "freeze_path": freeze,
        "artifacts_dir": artifacts,
        "output_dir": tmp_path / "final",
    }


def test_external_evaluation_is_hash_gated_and_complete(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)

    result = evaluate_frozen_models(**paths, repetitions=2)

    assert result["external_test_rows"] == 36
    assert set(result["conditions"]) == {"full", "reduced"}
    assert len(result["conditions"]["full"]["confusion_matrix"]) == 3
    assert (paths["output_dir"] / "metrics.json").is_file()
    assert (paths["output_dir"] / "timing_repetitions.csv").is_file()
    assert (paths["output_dir"] / "confusion_reduced.png").is_file()


def test_external_evaluation_refuses_overwrite(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)
    paths["output_dir"].mkdir()

    with pytest.raises(DataPreparationError, match="refusing overwrite"):
        evaluate_frozen_models(**paths, repetitions=1)
