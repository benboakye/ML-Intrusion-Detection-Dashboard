"""Dataset discovery, label inspection, hashing, and manifest generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Final, Iterable, Sequence

import pandas as pd

from ml_nids.config import CSV_CHUNK_SIZE, LABEL_COLUMN, TARGET_LABELS

DATASET_NAME: Final = "CIC-DDoS2019"
OFFICIAL_SOURCE_URL: Final = "https://www.unb.ca/cic/datasets/ddos-2019.html"
MANIFEST_FIELDS: Final = (
    "dataset",
    "official_source",
    "download_date",
    "official_day",
    "role",
    "local_file",
    "size_bytes",
    "row_count",
    "labels",
    "sha256",
)


class DatasetGovernanceError(RuntimeError):
    """Raised when source data cannot be safely inventoried."""


@dataclass(frozen=True)
class DatasetPartition:
    """User-confirmed mapping from an official collection day to a local role."""

    role: str
    official_day: str
    directory: Path
    download_date: date

    def __post_init__(self) -> None:
        if self.role not in {"training", "external_test"}:
            raise ValueError(f"Unsupported dataset role: {self.role}")
        if not self.official_day.strip():
            raise ValueError("official_day must not be blank")


@dataclass(frozen=True)
class ManifestRecord:
    dataset: str
    official_source: str
    download_date: str
    official_day: str
    role: str
    local_file: str
    size_bytes: int
    row_count: int
    labels: str
    sha256: str


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def discover_csv_files(directory: Path) -> list[Path]:
    """Find CSV files recursively in deterministic path order."""

    if not directory.is_dir():
        raise DatasetGovernanceError(f"Dataset directory does not exist: {directory}")
    files = sorted(
        (path for path in directory.rglob("*.csv") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix().casefold(),
    )
    if not files:
        raise DatasetGovernanceError(f"No CSV files found under: {directory}")
    return files


def _label_source_column(path: Path) -> str:
    try:
        header = pd.read_csv(path, nrows=0, low_memory=False)
    except Exception as exc:  # pandas exposes several parser-specific exceptions
        raise DatasetGovernanceError(f"Could not read CSV header: {path}") from exc

    normalized = [str(column).strip() for column in header.columns]
    if len(normalized) != len(set(normalized)):
        raise DatasetGovernanceError(
            f"Column names are duplicated after trimming whitespace: {path}"
        )
    if LABEL_COLUMN not in normalized:
        raise DatasetGovernanceError(f"Missing {LABEL_COLUMN!r} column: {path}")
    return str(header.columns[normalized.index(LABEL_COLUMN)])


def scan_labels(path: Path) -> tuple[int, tuple[str, ...]]:
    """Scan only the label column and return row count and normalized labels."""

    source_column = _label_source_column(path)
    labels: set[str] = set()
    row_count = 0
    try:
        chunks = pd.read_csv(
            path,
            usecols=[source_column],
            chunksize=CSV_CHUNK_SIZE,
            low_memory=False,
        )
        for chunk in chunks:
            row_count += len(chunk)
            values = chunk[source_column].dropna().astype(str).str.strip().str.upper()
            labels.update(value for value in values.unique() if value)
    except Exception as exc:
        raise DatasetGovernanceError(f"Could not scan labels in: {path}") from exc
    return row_count, tuple(sorted(labels))


def inspect_partition(partition: DatasetPartition) -> list[ManifestRecord]:
    """Create one manifest record per CSV in a partition."""

    records: list[ManifestRecord] = []
    for path in discover_csv_files(partition.directory):
        row_count, labels = scan_labels(path)
        records.append(
            ManifestRecord(
                dataset=DATASET_NAME,
                official_source=OFFICIAL_SOURCE_URL,
                download_date=partition.download_date.isoformat(),
                official_day=partition.official_day.strip(),
                role=partition.role,
                local_file=path.relative_to(partition.directory).as_posix(),
                size_bytes=path.stat().st_size,
                row_count=row_count,
                labels=";".join(labels),
                sha256=sha256_file(path),
            )
        )
    return records


def build_manifest(partitions: Sequence[DatasetPartition]) -> list[ManifestRecord]:
    """Inspect distinct training and external-test partitions."""

    roles = [partition.role for partition in partitions]
    if sorted(roles) != ["external_test", "training"]:
        raise ValueError("Provide exactly one training and one external_test partition")

    resolved = [partition.directory.resolve() for partition in partitions]
    if len(resolved) != len(set(resolved)):
        raise DatasetGovernanceError(
            "Training and external-test directories must be physically separate"
        )

    records: list[ManifestRecord] = []
    for partition in partitions:
        records.extend(inspect_partition(partition))
    return records


def missing_required_labels(
    records: Iterable[ManifestRecord],
) -> dict[str, tuple[str, ...]]:
    """Return target labels absent from each dataset role."""

    labels_by_role = {"training": set(), "external_test": set()}
    for record in records:
        labels_by_role[record.role].update(filter(None, record.labels.split(";")))
    required = set(TARGET_LABELS)
    return {
        role: tuple(sorted(required - observed))
        for role, observed in labels_by_role.items()
        if required - observed
    }


def write_manifest(records: Sequence[ManifestRecord], output: Path) -> None:
    """Write a deterministic UTF-8 CSV manifest."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory CIC-DDoS2019 CSVs without reading feature columns."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train_day"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/raw/test_day"))
    parser.add_argument("--train-official-day", required=True)
    parser.add_argument("--test-official-day", required=True)
    parser.add_argument("--download-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/dataset_manifest.csv")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    partitions = (
        DatasetPartition(
            role="training",
            official_day=args.train_official_day,
            directory=args.train_dir,
            download_date=args.download_date,
        ),
        DatasetPartition(
            role="external_test",
            official_day=args.test_official_day,
            directory=args.test_dir,
            download_date=args.download_date,
        ),
    )
    try:
        records = build_manifest(partitions)
        write_manifest(records, args.output)
    except (DatasetGovernanceError, ValueError) as exc:
        parser.error(str(exc))

    for role in ("training", "external_test"):
        role_records = [record for record in records if record.role == role]
        rows = sum(record.row_count for record in role_records)
        print(f"{role}: {len(role_records)} CSV file(s), {rows:,} rows")
    print(f"manifest: {args.output}")

    missing = missing_required_labels(records)
    if missing:
        for role, labels in missing.items():
            print(f"{role} is missing required labels: {', '.join(labels)}", file=sys.stderr)
        return 2
    return 0
