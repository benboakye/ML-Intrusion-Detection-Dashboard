import csv
import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ml_nids.governance import (
    DatasetGovernanceError,
    DatasetPartition,
    build_manifest,
    missing_required_labels,
    scan_labels,
    write_manifest,
)


def _write_csv(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({" Flow Duration": range(len(labels)), " Label": labels}).to_csv(
        path, index=False
    )


def _partitions(tmp_path: Path) -> tuple[DatasetPartition, DatasetPartition]:
    acquired = date(2026, 9, 1)
    return (
        DatasetPartition("training", "confirmed training day", tmp_path / "train", acquired),
        DatasetPartition(
            "external_test", "confirmed testing day", tmp_path / "test", acquired
        ),
    )


def test_scan_labels_reads_only_normalized_label_values(tmp_path: Path) -> None:
    source = tmp_path / "flows.csv"
    _write_csv(source, [" benign ", "Syn", "UDP", "SYN"])

    row_count, labels = scan_labels(source)

    assert row_count == 4
    assert labels == ("BENIGN", "SYN", "UDP")


def test_manifest_is_deterministic_and_contains_hashes(tmp_path: Path) -> None:
    train, test = _partitions(tmp_path)
    _write_csv(train.directory / "b.csv", ["SYN", "UDP"])
    _write_csv(train.directory / "a.csv", ["BENIGN"])
    _write_csv(test.directory / "flows.csv", ["BENIGN", "SYN", "UDP"])

    records = build_manifest((train, test))
    output = tmp_path / "manifest.csv"
    write_manifest(records, output)

    assert [record.local_file for record in records] == ["a.csv", "b.csv", "flows.csv"]
    assert missing_required_labels(records) == {}
    expected_hash = hashlib.sha256((train.directory / "a.csv").read_bytes()).hexdigest()
    assert records[0].sha256 == expected_hash
    with output.open(encoding="utf-8", newline="") as source:
        written = list(csv.DictReader(source))
    assert len(written) == 3
    assert written[0]["official_source"].startswith("https://www.unb.ca/")


def test_missing_required_labels_are_reported_by_role(tmp_path: Path) -> None:
    train, test = _partitions(tmp_path)
    _write_csv(train.directory / "flows.csv", ["BENIGN", "SYN"])
    _write_csv(test.directory / "flows.csv", ["BENIGN", "UDP"])

    missing = missing_required_labels(build_manifest((train, test)))

    assert missing == {"training": ("UDP",), "external_test": ("SYN",)}


def test_training_udp_alias_is_explicit_and_raw_label_is_preserved(
    tmp_path: Path,
) -> None:
    train, test = _partitions(tmp_path)
    _write_csv(train.directory / "flows.csv", ["BENIGN", "SYN", "DrDoS_UDP"])
    _write_csv(test.directory / "flows.csv", ["BENIGN", "SYN", "UDP", "MSSQL"])

    records = build_manifest((train, test))

    assert records[0].labels == "BENIGN;DRDOS_UDP;SYN"
    assert records[0].effective_labels == "BENIGN;SYN;UDP"
    assert records[1].labels == "BENIGN;MSSQL;SYN;UDP"
    assert records[1].effective_labels == "BENIGN;SYN;UDP"
    assert missing_required_labels(records) == {}


def test_partitions_must_be_physically_separate(tmp_path: Path) -> None:
    acquired = date(2026, 9, 1)
    shared = tmp_path / "shared"
    _write_csv(shared / "flows.csv", ["BENIGN", "SYN", "UDP"])

    with pytest.raises(DatasetGovernanceError, match="physically separate"):
        build_manifest(
            (
                DatasetPartition("training", "day one", shared, acquired),
                DatasetPartition("external_test", "day two", shared, acquired),
            )
        )


def test_duplicate_columns_after_trimming_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.csv"
    source.write_text("Label, Label\nBENIGN,SYN\n", encoding="utf-8")

    with pytest.raises(DatasetGovernanceError, match="duplicated"):
        scan_labels(source)
