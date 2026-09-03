"""Rank prepared training features and freeze the reduced feature set."""

from ml_nids.analysis import select_training_features
from ml_nids.config import ARTIFACTS_DIR, PROCESSED_DIR


if __name__ == "__main__":
    ranking, selected, metadata = select_training_features(
        PROCESSED_DIR / "train.parquet",
        ranking_path=ARTIFACTS_DIR / "mi_ranking.csv",
        selected_path=ARTIFACTS_DIR / "selected_features.json",
        metadata_path=ARTIFACTS_DIR / "feature_selection_summary.json",
    )
    print(f"ranked features: {len(ranking)}")
    print(f"selected features: {len(selected)}")
    print(f"external test used: {metadata['external_test_used']}")
