import json
from pathlib import Path

import pandas as pd

from ml_nids.config import LABEL_COLUMN
from ml_nids.preparation import collect_partition, prepare_dataset, prepare_frames


def _write_source(path: Path, labels: list[str], offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = len(labels)
    pd.DataFrame(
        {
            " Flow ID": [f"flow-{offset + index}" for index in range(size)],
            " Protocol": [6 if "SYN" in label.upper() else 17 for label in labels],
            " Feature A": range(offset, offset + size),
            " Constant": [1] * size,
            " Label": labels,
        }
    ).to_csv(path, index=False)


def test_collection_applies_only_the_governed_alias_and_caps_classes(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    _write_source(
        train_dir / "flows.csv",
        ["BENIGN"] * 4 + ["SYN"] * 4 + ["DrDoS_UDP"] * 4 + ["UDP-lag"],
    )

    result = collect_partition(train_dir, "training", 3, chunk_size=2)

    assert result.frame[LABEL_COLUMN].value_counts().to_dict() == {
        "BENIGN": 3,
        "SYN": 3,
        "UDP": 3,
    }
    assert result.source_label_counts["DRDOS_UDP"] == 4
    assert result.effective_target_counts["UDP"] == 4
    assert result.excluded_source_counts == {"UDP-LAG": 1}


def test_preparation_uses_shared_training_varying_features_only(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _write_source(
        train_dir / "flows.csv",
        ["BENIGN"] * 3 + ["SYN"] * 3 + ["DrDoS_UDP"] * 3,
    )
    _write_source(
        test_dir / "flows.csv",
        ["BENIGN"] * 3 + ["SYN"] * 3 + ["UDP"] * 3 + ["MSSQL"],
        offset=100,
    )

    training = collect_partition(train_dir, "training", 2, chunk_size=2)
    external_test = collect_partition(test_dir, "external_test", 2, chunk_size=2)
    prepared = prepare_frames(training, external_test)

    assert prepared.usable_features == ("Feature A",)
    assert list(prepared.training.columns) == ["Feature A", LABEL_COLUMN]
    assert set(prepared.external_test[LABEL_COLUMN]) == {"BENIGN", "SYN", "UDP"}
    assert prepared.summary["external_test"]["excluded_source_counts"] == {
        "MSSQL": 1
    }


def test_collection_balances_down_to_a_scarce_class_without_duplication(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    _write_source(
        train_dir / "flows.csv",
        ["BENIGN"] * 2 + ["SYN"] * 8 + ["DrDoS_UDP"] * 8,
    )

    result = collect_partition(train_dir, "training", 5, chunk_size=3)

    assert result.frame[LABEL_COLUMN].value_counts().to_dict() == {
        "BENIGN": 2,
        "SYN": 2,
        "UDP": 2,
    }


def test_prepare_dataset_writes_parquet_and_audit_files(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _write_source(
        train_dir / "flows.csv",
        ["BENIGN"] * 2 + ["SYN"] * 2 + ["DrDoS_UDP"] * 2,
    )
    _write_source(
        test_dir / "flows.csv", ["BENIGN"] * 2 + ["SYN"] * 2 + ["UDP"] * 2
    )
    processed = tmp_path / "processed"
    summary = tmp_path / "summary.json"
    features = tmp_path / "features.json"

    result = prepare_dataset(
        train_dir=train_dir,
        test_dir=test_dir,
        processed_dir=processed,
        summary_path=summary,
        features_path=features,
        train_cap=2,
        test_cap=2,
        chunk_size=2,
    )

    assert len(pd.read_parquet(processed / "train.parquet")) == 6
    assert len(pd.read_parquet(processed / "external_test.parquet")) == 6
    assert json.loads(features.read_text(encoding="utf-8")) == ["Feature A"]
    audit = json.loads(summary.read_text(encoding="utf-8"))
    assert audit["usable_feature_count"] == 1
    assert audit["training"]["requested_class_cap"] == 2
    assert len(audit["outputs"]["training"]["sha256"]) == 64
    assert result.usable_features == ("Feature A",)
