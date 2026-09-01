# Results and evidence

Important runs use a new dated directory and are never overwritten after they
are referenced in the report.

Example:

```text
results/
|-- 2026-09-15_run01/
|   |-- config.json
|   |-- validation_metrics.csv
|   `-- notes.txt
`-- final_external_test/
    |-- metrics.csv
    |-- confusion_matrices.png
    `-- benchmark.csv
```

Each run must record its purpose, code version, environment lock hash, data
hashes, seed, feature condition, model settings, row counts, outputs, unexpected
events, and resulting decision.
