"""Memory-bounded preparation of the governed CIC-DDoS2019 partitions."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_nids.config import (
    CSV_CHUNK_SIZE,
    EXTERNAL_TEST_CLASS_CAP,
    FORBIDDEN_FEATURES,
    LABEL_ALIASES_BY_ROLE,
    LABEL_COLUMN,
    PROCESSED_DIR,
    RAW_TEST_DIR,
    RAW_TRAIN_DIR,
    SEED,
    TARGET_LABELS,
    TRAIN_CLASS_CAP,
)
from ml_nids.governance import DatasetGovernanceError, discover_csv_files, sha256_file
from ml_nids.labels import normalize_label_series, validate_role

_PRIORITY_COLUMN = "__governed_sample_priority__"


class DataPreparationError(RuntimeError):
    """Raised when raw data cannot satisfy the frozen preparation contract."""


@dataclass(frozen=True)
class CollectedPartition:
    """A capped partition plus source-label evidence from the complete scan."""

    frame: pd.DataFrame
    source_label_counts: dict[str, int]
    effective_target_counts: dict[str, int]
    excluded_source_counts: dict[str, int]


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared frames and the evidence needed to reproduce them."""

    training: pd.DataFrame
    external_test: pd.DataFrame
    usable_features: tuple[str, ...]
    summary: dict[str, Any]


def _normalize_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    columns = [str(column).strip() for column in frame.columns]
    if len(columns) != len(set(columns)):
        raise DataPreparationError(
            f"Column names are duplicated after trimming whitespace: {path}"
        )
    frame.columns = columns
    if LABEL_COLUMN not in frame:
        raise DataPreparationError(f"Missing {LABEL_COLUMN!r} column: {path}")
    if _PRIORITY_COLUMN in frame:
        raise DataPreparationError(
            f"Reserved column {_PRIORITY_COLUMN!r} exists in source: {path}"
        )
    return frame


def collect_partition(
    directory: Path,
    role: str,
    class_cap: int,
    *,
    chunk_size: int = CSV_CHUNK_SIZE,
    seed: int = SEED,
) -> CollectedPartition:
    """Scan a role and retain a deterministic uniform sample per target class."""

    validate_role(role)
    if class_cap <= 0 or chunk_size <= 0:
        raise ValueError("class_cap and chunk_size must be positive")

    try:
        paths = discover_csv_files(directory)
    except DatasetGovernanceError as exc:
        raise DataPreparationError(str(exc)) from exc

    rng = np.random.default_rng(seed)
    buckets = {label: pd.DataFrame() for label in TARGET_LABELS}
    source_counts: Counter[str] = Counter()
    effective_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()

    for path in paths:
        try:
            chunks = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
            for chunk in chunks:
                chunk = _normalize_columns(chunk, path)
                raw_labels = chunk[LABEL_COLUMN].astype("string").str.strip().str.upper()
                effective_labels = normalize_label_series(chunk[LABEL_COLUMN], role)

                source_counts.update(raw_labels.dropna().tolist())
                target_mask = effective_labels.isin(TARGET_LABELS)
                effective_counts.update(effective_labels[target_mask].dropna().tolist())
                excluded_counts.update(raw_labels[~target_mask].dropna().tolist())

                for label in TARGET_LABELS:
                    part = chunk.loc[effective_labels.eq(label)].copy()
                    if part.empty:
                        continue
                    part[LABEL_COLUMN] = label
                    part[_PRIORITY_COLUMN] = rng.random(len(part))
                    combined = pd.concat((buckets[label], part), ignore_index=True)
                    if len(combined) > class_cap:
                        combined = combined.nsmallest(class_cap, _PRIORITY_COLUMN)
                    buckets[label] = combined
        except DataPreparationError:
            raise
        except Exception as exc:
            raise DataPreparationError(f"Could not prepare source CSV: {path}") from exc

    missing = [label for label, frame in buckets.items() if frame.empty]
    if missing:
        raise DataPreparationError(
            f"{role} contains no usable rows for: {', '.join(missing)}"
        )

    # Caps are maxima, not a license to duplicate scarce records. Balance all
    # classes down to the smallest available target class when necessary.
    applied_cap = min(class_cap, *(len(frame) for frame in buckets.values()))
    for label, frame in buckets.items():
        if len(frame) > applied_cap:
            buckets[label] = frame.nsmallest(applied_cap, _PRIORITY_COLUMN)

    selected = pd.concat(
        [buckets[label] for label in TARGET_LABELS], ignore_index=True
    ).drop(columns=_PRIORITY_COLUMN)
    selected = selected.sample(frac=1, random_state=seed).reset_index(drop=True)
    return CollectedPartition(
        frame=selected,
        source_label_counts=dict(sorted(source_counts.items())),
        effective_target_counts=dict(sorted(effective_counts.items())),
        excluded_source_counts=dict(sorted(excluded_counts.items())),
    )


def _numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=[LABEL_COLUMN, *FORBIDDEN_FEATURES], errors="ignore")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def prepare_frames(
    training: CollectedPartition, external_test: CollectedPartition
) -> PreparedDataset:
    """Clean selected rows using training-only feature usability decisions."""

    train_features = _numeric_features(training.frame)
    test_features = _numeric_features(external_test.frame)
    shared = [column for column in train_features if column in test_features]
    usable = tuple(
        column
        for column in shared
        if train_features[column].nunique(dropna=True) > 1
    )
    if not usable:
        raise DataPreparationError("No usable shared numeric features remain")

    train_output = train_features.loc[:, usable].copy()
    test_output = test_features.loc[:, usable].copy()
    train_output[LABEL_COLUMN] = training.frame[LABEL_COLUMN].to_numpy()
    test_output[LABEL_COLUMN] = external_test.frame[LABEL_COLUMN].to_numpy()

    summary: dict[str, Any] = {
        "seed": SEED,
        "chunk_size": CSV_CHUNK_SIZE,
        "label_policy": {
            role: dict(aliases) for role, aliases in LABEL_ALIASES_BY_ROLE.items()
        },
        "forbidden_features_removed": sorted(
            set(FORBIDDEN_FEATURES)
            & (set(training.frame.columns) | set(external_test.frame.columns))
        ),
        "usable_feature_count": len(usable),
        "training": {
            "selected_rows": len(train_output),
            "applied_class_size": int(
                train_output[LABEL_COLUMN].value_counts().min()
            ),
            "selected_class_counts": dict(
                sorted(train_output[LABEL_COLUMN].value_counts().to_dict().items())
            ),
            "source_label_counts": training.source_label_counts,
            "effective_target_counts": training.effective_target_counts,
            "excluded_source_counts": training.excluded_source_counts,
        },
        "external_test": {
            "selected_rows": len(test_output),
            "applied_class_size": int(
                test_output[LABEL_COLUMN].value_counts().min()
            ),
            "selected_class_counts": dict(
                sorted(test_output[LABEL_COLUMN].value_counts().to_dict().items())
            ),
            "source_label_counts": external_test.source_label_counts,
            "effective_target_counts": external_test.effective_target_counts,
            "excluded_source_counts": external_test.excluded_source_counts,
        },
    }
    return PreparedDataset(train_output, test_output, usable, summary)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="snappy")
    os.replace(temporary, path)


def _write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_dataset(
    *,
    train_dir: Path = RAW_TRAIN_DIR,
    test_dir: Path = RAW_TEST_DIR,
    processed_dir: Path = PROCESSED_DIR,
    summary_path: Path,
    features_path: Path,
    train_cap: int = TRAIN_CLASS_CAP,
    test_cap: int = EXTERNAL_TEST_CLASS_CAP,
    chunk_size: int = CSV_CHUNK_SIZE,
) -> PreparedDataset:
    """Prepare both partitions and persist reproducible outputs atomically."""

    training = collect_partition(
        train_dir, "training", train_cap, chunk_size=chunk_size
    )
    external_test = collect_partition(
        test_dir, "external_test", test_cap, chunk_size=chunk_size
    )
    prepared = prepare_frames(training, external_test)

    train_output = processed_dir / "train.parquet"
    test_output = processed_dir / "external_test.parquet"
    _write_parquet_atomic(prepared.training, train_output)
    _write_parquet_atomic(prepared.external_test, test_output)
    prepared.summary["chunk_size"] = chunk_size
    prepared.summary["training"]["requested_class_cap"] = train_cap
    prepared.summary["external_test"]["requested_class_cap"] = test_cap
    prepared.summary["outputs"] = {
        "external_test": {
            "file": test_output.name,
            "sha256": sha256_file(test_output),
            "size_bytes": test_output.stat().st_size,
        },
        "training": {
            "file": train_output.name,
            "sha256": sha256_file(train_output),
            "size_bytes": train_output.stat().st_size,
        },
    }
    _write_json_atomic(list(prepared.usable_features), features_path)
    _write_json_atomic(prepared.summary, summary_path)
    return prepared
