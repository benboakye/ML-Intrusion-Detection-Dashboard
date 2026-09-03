"""Prepare balanced training and independent test Parquet datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ml_nids.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_TEST_DIR, RAW_TRAIN_DIR
from ml_nids.preparation import DataPreparationError, prepare_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare governed CIC-DDoS2019 samples in bounded memory."
    )
    parser.add_argument("--train-dir", type=Path, default=RAW_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=RAW_TEST_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ARTIFACTS_DIR / "data_preparation_summary.json",
    )
    parser.add_argument(
        "--features", type=Path, default=ARTIFACTS_DIR / "usable_features.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prepared = prepare_dataset(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            processed_dir=args.processed_dir,
            summary_path=args.summary,
            features_path=args.features,
        )
    except (DataPreparationError, ValueError) as exc:
        _parser().error(str(exc))

    print(
        f"training: {len(prepared.training):,} rows; "
        f"external_test: {len(prepared.external_test):,} rows"
    )
    print(f"usable features: {len(prepared.usable_features)}")
    print(f"processed data: {args.processed_dir}")
    print(f"summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
