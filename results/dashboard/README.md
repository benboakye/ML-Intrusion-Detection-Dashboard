# Dashboard evidence

This directory records the controlled-replay functional test of the frozen
reduced model using the prepared 1,500-row sample described by
`artifacts/replay_manifest.json`.

- `controlled_replay.png` shows validation, the permanent research warning,
  model condition, safety limit, and headline replay counts.
- `controlled_replay_results.png` shows the class-count chart and recent-alert
  table.
- `replay_predictions.csv` is the row-level prediction log generated through
  the same replay functions used by the dashboard.
- `replay_run.json` records the input/model identifiers, output hash, counts,
  batch size, and measured performance for the saved run.

The evidence preserves the known negative result: only two of the prepared SYN
records were predicted as SYN. It must not be interpreted as operational
detector validation.

