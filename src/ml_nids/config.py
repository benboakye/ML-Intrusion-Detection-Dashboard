"""Frozen experiment constants and repository paths.

Dataset-dependent scripts import their decisions from this module so the full
and reduced model conditions cannot silently drift apart.
"""

from pathlib import Path
from types import MappingProxyType
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DATA_DIR: Final = PROJECT_ROOT / "data"
RAW_TRAIN_DIR: Final = DATA_DIR / "raw" / "train_day"
RAW_TEST_DIR: Final = DATA_DIR / "raw" / "test_day"
INTERIM_DIR: Final = DATA_DIR / "interim"
PROCESSED_DIR: Final = DATA_DIR / "processed"
REPLAY_DIR: Final = DATA_DIR / "replay"
ARTIFACTS_DIR: Final = PROJECT_ROOT / "artifacts"
RESULTS_DIR: Final = PROJECT_ROOT / "results"

SEED: Final = 42
LABEL_COLUMN: Final = "Label"
TARGET_LABELS: Final = ("BENIGN", "SYN", "UDP")

CSV_CHUNK_SIZE: Final = 50_000
TRAIN_CLASS_CAP: Final = 50_000
EXTERNAL_TEST_CLASS_CAP: Final = 20_000
FEATURE_RANKING_CLASS_CAP: Final = 10_000
REDUCED_FEATURE_COUNT: Final = 15

FORBIDDEN_FEATURES: Final = frozenset(
    {
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Source Port",
        "Destination Port",
        "Protocol",
        "SimillarHTTP",
        "Inbound",
    }
)

RANDOM_FOREST_PARAMS: Final = MappingProxyType(
    {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "max_samples": 0.7,
        "class_weight": "balanced_subsample",
        "n_jobs": 2,
        "random_state": SEED,
    }
)

PRIMARY_METRIC: Final = "macro_f1"
MAX_MACRO_F1_LOSS: Final = 0.01
MAX_ATTACK_RECALL_LOSS: Final = 0.02
BENCHMARK_REPETITIONS: Final = 5


def required_directories() -> tuple[Path, ...]:
    """Return the directories created by dataset-dependent pipeline stages."""

    return (
        RAW_TRAIN_DIR,
        RAW_TEST_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        REPLAY_DIR,
        ARTIFACTS_DIR,
        RESULTS_DIR,
    )
