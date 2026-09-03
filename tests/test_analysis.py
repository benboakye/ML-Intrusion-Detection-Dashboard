import json
from pathlib import Path

import pandas as pd
import pytest

from ml_nids.analysis import (
    load_training_data,
    rank_features,
    select_training_features,
    write_exploration,
)
from ml_nids.preparation import DataPreparationError


def _training_frame(rows_per_class: int = 8) -> pd.DataFrame:
    rows = []
    for class_number, label in enumerate(("BENIGN", "SYN", "UDP")):
        for index in range(rows_per_class):
            rows.append(
                {
                    "signal": class_number * 100 + index,
                    "noise": (index * 7 + class_number) % 11,
                    "with_missing": None if index % 3 == 0 else index,
                    "Label": label,
                }
            )
    frame = pd.DataFrame(rows)
    for feature_number in range(13):
        frame[f"extra_{feature_number:02d}"] = (
            frame.index * (feature_number + 2) + feature_number
        ) % 17
    return frame


def test_training_loader_rejects_incomplete_labels(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _training_frame().query("Label != 'UDP'").to_parquet(path, index=False)

    with pytest.raises(DataPreparationError, match="expected"):
        load_training_data(path)


def test_feature_ranking_is_deterministic_and_training_only() -> None:
    frame = _training_frame()

    first = rank_features(frame, sample_cap=5, selected_count=2)
    second = rank_features(frame, sample_cap=5, selected_count=2)

    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[1] == second[1]
    assert len(first[1]) == len(set(first[1])) == 2
    assert first[2]["sampled_class_counts"] == {
        "BENIGN": 5,
        "SYN": 5,
        "UDP": 5,
    }
    assert first[2]["external_test_used"] is False


def test_analysis_outputs_are_complete(tmp_path: Path) -> None:
    training_path = tmp_path / "train.parquet"
    frame = _training_frame()
    frame.to_parquet(training_path, index=False)

    eda_outputs = write_exploration(frame, tmp_path / "eda")
    ranking, selected, metadata = select_training_features(
        training_path,
        ranking_path=tmp_path / "mi_ranking.csv",
        selected_path=tmp_path / "selected_features.json",
        metadata_path=tmp_path / "feature_selection_summary.json",
    )

    assert all(path.is_file() and path.stat().st_size > 0 for path in eda_outputs)
    assert len(ranking) == 16
    assert len(selected) == 15
    assert len(metadata["training_sha256"]) == 64
    assert json.loads((tmp_path / "selected_features.json").read_text()) == list(
        selected
    )
