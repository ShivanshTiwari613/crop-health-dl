Every subfolder here is one training run produced by `python -m src.train --config configs/<name>.yaml`:

- `config.yaml` — exact config used
- `history.csv` — per-epoch train/val loss + accuracy, by phase
- `metrics.json` — test-split accuracy, macro/weighted F1, precision, recall, params, wall time
- `classification_report.txt`, `confusion_matrix.{png,csv}`

Model weights (`model.keras`) are git-ignored. `RESULTS.md` is regenerated with `make table`.
