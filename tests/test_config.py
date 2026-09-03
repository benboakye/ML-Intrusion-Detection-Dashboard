from ml_nids import __version__
from ml_nids.config import (
    BENCHMARK_REPETITIONS,
    EXTERNAL_TEST_CLASS_CAP,
    FEATURE_RANKING_CLASS_CAP,
    FORBIDDEN_FEATURES,
    LABEL_ALIASES_BY_ROLE,
    MAX_ATTACK_RECALL_LOSS,
    MAX_MACRO_F1_LOSS,
    PROJECT_ROOT,
    RANDOM_FOREST_PARAMS,
    REDUCED_FEATURE_COUNT,
    SEED,
    TARGET_LABELS,
    TRAIN_CLASS_CAP,
    required_directories,
)


def test_package_version_is_initialized() -> None:
    assert __version__ == "0.1.0"


def test_fixed_experiment_decisions() -> None:
    assert SEED == 42
    assert TARGET_LABELS == ("BENIGN", "SYN", "UDP")
    assert TRAIN_CLASS_CAP == 50_000
    assert EXTERNAL_TEST_CLASS_CAP == 20_000
    assert FEATURE_RANKING_CLASS_CAP == 10_000
    assert REDUCED_FEATURE_COUNT == 15
    assert BENCHMARK_REPETITIONS == 5
    assert MAX_MACRO_F1_LOSS == 0.01
    assert MAX_ATTACK_RECALL_LOSS == 0.02
    assert dict(LABEL_ALIASES_BY_ROLE["training"]) == {"DRDOS_UDP": "UDP"}
    assert dict(LABEL_ALIASES_BY_ROLE["external_test"]) == {}


def test_shortcut_and_identifier_fields_are_forbidden() -> None:
    expected = {
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Source Port",
        "Destination Port",
        "Protocol",
        "Unnamed: 0",
        "Fwd Header Length.1",
    }
    assert expected <= FORBIDDEN_FEATURES


def test_random_forest_settings_are_resource_bounded_and_reproducible() -> None:
    assert RANDOM_FOREST_PARAMS["n_estimators"] == 100
    assert RANDOM_FOREST_PARAMS["max_depth"] == 20
    assert RANDOM_FOREST_PARAMS["n_jobs"] == 2
    assert RANDOM_FOREST_PARAMS["random_state"] == SEED


def test_all_managed_paths_remain_inside_project() -> None:
    assert (PROJECT_ROOT / "README.md").is_file()
    assert all(path.is_relative_to(PROJECT_ROOT) for path in required_directories())
