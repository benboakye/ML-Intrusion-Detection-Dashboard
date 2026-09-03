"""Training-only exploratory evidence and mutual-information feature ranking."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from ml_nids.config import (
    FEATURE_RANKING_CLASS_CAP,
    LABEL_COLUMN,
    PROJECT_ROOT,
    REDUCED_FEATURE_COUNT,
    SEED,
    TARGET_LABELS,
)
from ml_nids.governance import sha256_file
from ml_nids.preparation import DataPreparationError


def load_training_data(path: Path) -> pd.DataFrame:
    """Load and validate the prepared training partition only."""

    if not path.is_file():
        raise DataPreparationError(f"Prepared training data does not exist: {path}")
    frame = pd.read_parquet(path)
    if LABEL_COLUMN not in frame:
        raise DataPreparationError(f"Missing {LABEL_COLUMN!r} column: {path}")
    if frame.columns.duplicated().any():
        raise DataPreparationError(f"Prepared columns are duplicated: {path}")
    observed = set(frame[LABEL_COLUMN].dropna().astype(str))
    if observed != set(TARGET_LABELS):
        raise DataPreparationError(
            f"Prepared training labels are {sorted(observed)}, expected {list(TARGET_LABELS)}"
        )
    non_numeric = frame.drop(columns=LABEL_COLUMN).select_dtypes(exclude="number")
    if not non_numeric.empty:
        raise DataPreparationError(
            f"Prepared training features are not numeric: {list(non_numeric.columns)}"
        )
    return frame


def write_exploration(frame: pd.DataFrame, output_dir: Path) -> tuple[Path, ...]:
    """Write compact training-only EDA tables and a class-count plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = (
        frame[LABEL_COLUMN]
        .value_counts()
        .reindex(TARGET_LABELS, fill_value=0)
        .rename("count")
    )
    features = frame.drop(columns=LABEL_COLUMN)
    missingness = features.isna().mean().sort_values(ascending=False).rename(
        "missing_fraction"
    )
    summary = features.describe().transpose()

    class_counts_path = output_dir / "class_counts.csv"
    missingness_path = output_dir / "missingness.csv"
    summary_path = output_dir / "summary_statistics.csv"
    plot_path = output_dir / "class_counts.png"
    class_counts.to_csv(class_counts_path)
    missingness.to_csv(missingness_path)
    summary.to_csv(summary_path)

    os.environ.setdefault(
        "MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim" / "matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.bar(class_counts.index, class_counts.values, color="#4c78a8")
    axis.set(title="Prepared training class counts", xlabel="Class", ylabel="Rows")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return class_counts_path, missingness_path, summary_path, plot_path


def rank_features(
    frame: pd.DataFrame,
    *,
    sample_cap: int = FEATURE_RANKING_CLASS_CAP,
    selected_count: int = REDUCED_FEATURE_COUNT,
    seed: int = SEED,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    """Rank numeric features using a balanced sample from training data only."""

    if sample_cap <= 0 or selected_count <= 0:
        raise ValueError("sample_cap and selected_count must be positive")
    features = frame.drop(columns=LABEL_COLUMN)
    if selected_count > features.shape[1]:
        raise DataPreparationError(
            f"Requested {selected_count} features but only {features.shape[1]} are usable"
        )

    sampled_parts = []
    sampled_counts: dict[str, int] = {}
    for label in TARGET_LABELS:
        group = frame.loc[frame[LABEL_COLUMN].eq(label)]
        size = min(len(group), sample_cap)
        sampled_parts.append(group.sample(n=size, random_state=seed))
        sampled_counts[label] = size
    sample = pd.concat(sampled_parts, ignore_index=True).sample(
        frac=1, random_state=seed
    )
    labels = sample.pop(LABEL_COLUMN)
    imputed = SimpleImputer(strategy="median").fit_transform(sample)
    scores = mutual_info_classif(imputed, labels, random_state=seed)
    ranking = (
        pd.DataFrame({"feature": sample.columns, "mi_score": scores})
        .sort_values(["mi_score", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selected = tuple(ranking.head(selected_count)["feature"])
    metadata: dict[str, Any] = {
        "method": "mutual_info_classif",
        "seed": seed,
        "sample_cap_per_class": sample_cap,
        "sampled_class_counts": sampled_counts,
        "input_feature_count": features.shape[1],
        "selected_feature_count": len(selected),
        "external_test_used": False,
    }
    return ranking, selected, metadata


def write_feature_selection(
    ranking: pd.DataFrame,
    selected: tuple[str, ...],
    metadata: dict[str, Any],
    *,
    ranking_path: Path,
    selected_path: Path,
    metadata_path: Path,
) -> None:
    """Persist the complete ranking, frozen feature list, and audit metadata."""

    for path in (ranking_path, selected_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(ranking_path, index=False)
    selected_path.write_text(
        json.dumps(list(selected), indent=2) + "\n", encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def select_training_features(
    training_path: Path,
    *,
    ranking_path: Path,
    selected_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    """Run and persist the frozen training-only feature-selection stage."""

    frame = load_training_data(training_path)
    ranking, selected, metadata = rank_features(frame)
    metadata["training_file"] = training_path.name
    metadata["training_sha256"] = sha256_file(training_path)
    write_feature_selection(
        ranking,
        selected,
        metadata,
        ranking_path=ranking_path,
        selected_path=selected_path,
        metadata_path=metadata_path,
    )
    return ranking, selected, metadata
