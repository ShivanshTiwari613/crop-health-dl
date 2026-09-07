"""Collect results/*/metrics.json into a markdown table (results/RESULTS.md)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "run_name",
    "task",
    "img_size",
    "epochs",
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "train_seconds",
]


def main():
    runs = []
    for path in sorted(ROOT.glob("results/*/metrics.json")):
        if path.parent.name == "smoke":
            continue
        with open(path) as f:
            run = json.load(f)
        with open(path.parent / "history.csv") as f:
            run["epochs"] = sum(1 for _ in f) - 1
        runs.append(run)
    if not runs:
        print("no runs found under results/")
        return

    header = "| " + " | ".join(COLUMNS) + " |"
    rule = "|" + "---|" * len(COLUMNS)
    rows = ["| " + " | ".join(str(run.get(c, "")) for c in COLUMNS) + " |" for run in runs]
    table = "\n".join([header, rule, *rows]) + "\n"

    (ROOT / "results" / "RESULTS.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
