"""Create a small deterministic controlled-replay input and separate truth file."""

from ml_nids.config import ARTIFACTS_DIR, PROCESSED_DIR, REPLAY_DIR
from ml_nids.replay import make_replay_sample


if __name__ == "__main__":
    manifest = make_replay_sample(
        PROCESSED_DIR / "external_test.parquet",
        input_path=REPLAY_DIR / "replay_input.csv",
        truth_path=REPLAY_DIR / "replay_truth.csv",
        manifest_path=ARTIFACTS_DIR / "replay_manifest.json",
    )
    print(f"replay rows: {manifest['rows']}")
    print(f"class counts: {manifest['class_counts']}")
    print("truth is stored separately from dashboard input")
