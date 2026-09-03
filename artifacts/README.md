# Artifact workspace

This directory will contain small, reproducible metadata such as the complete
mutual-information ranking, the frozen 15-feature list, model configuration,
and approved artifact hashes.

The data-preparation stage writes `data_preparation_summary.json`, including
the label policy, exclusions, class counts, and processed-data hashes, plus
`usable_features.json`, the ordered shared numeric feature list. These small
audit artifacts are committed; the corresponding Parquet data is not.

Serialized models are ignored by Git by default. Python `joblib` and pickle
files can execute code when loaded and must be treated as trusted local
artifacts only.
