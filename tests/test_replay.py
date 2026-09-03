import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ml_nids.replay import (
    make_replay_sample,
    run_controlled_replay,
    validate_replay_frame,
)
from ml_nids.preparation import DataPreparationError


def _external_frame(rows_per_class: int = 6) -> pd.DataFrame:
    rows = []
    for number, label in enumerate(("BENIGN", "SYN", "UDP")):
        for index in range(rows_per_class):
            rows.append({"feature_a": number + index, "feature_b": index, "Label": label})
    return pd.DataFrame(rows)


def test_replay_sample_keeps_truth_separate(tmp_path: Path) -> None:
    external = tmp_path / "external.parquet"
    input_path = tmp_path / "replay_input.csv"
    truth_path = tmp_path / "replay_truth.csv"
    manifest_path = tmp_path / "replay_manifest.json"
    _external_frame().to_parquet(external, index=False)

    manifest = make_replay_sample(
        external,
        input_path=input_path,
        truth_path=truth_path,
        manifest_path=manifest_path,
        rows_per_class=3,
    )

    assert manifest["rows"] == 9
    assert "Label" not in pd.read_csv(input_path)
    assert list(pd.read_csv(truth_path).columns) == ["replay_row", "true_label"]
    assert json.loads(manifest_path.read_text())["input"]["contains_label"] is False


def test_replay_validation_rejects_missing_features_and_large_uploads() -> None:
    frame = pd.DataFrame({"feature_a": [1, 2]})
    with pytest.raises(DataPreparationError, match="missing"):
        validate_replay_frame(frame, ["feature_a", "feature_b"])
    with pytest.raises(DataPreparationError, match="maximum"):
        validate_replay_frame(frame, ["feature_a"], maximum_rows=1)


def test_controlled_replay_returns_predictions_and_timing() -> None:
    training = pd.DataFrame({"feature_a": [0, 1, 2], "feature_b": [1, 1, 1]})
    labels = ["BENIGN", "SYN", "UDP"]
    pipeline = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
    pipeline.fit(training, labels)
    bundle = {"pipeline": pipeline, "features": list(training.columns)}

    result, performance = run_controlled_replay(bundle, training, batch_size=2)

    assert len(result) == 3
    assert list(result.columns) == [
        "replay_row",
        "prediction",
        "batch_latency_seconds",
        "alert",
    ]
    assert performance["elapsed_seconds"] > 0
