# Artifact workspace

This directory will contain small, reproducible metadata such as the complete
mutual-information ranking, the frozen 15-feature list, model configuration,
and approved artifact hashes.

Serialized models are ignored by Git by default. Python `joblib` and pickle
files can execute code when loaded and must be treated as trusted local
artifacts only.
