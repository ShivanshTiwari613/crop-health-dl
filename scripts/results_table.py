"""Collect results/*/metrics.json into a markdown table (results/RESULTS.md)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "run_name",
    "backbone",
    "task",
    "img_size",
    "n_test",
    "accuracy",
    "f1_macro",
    "f1_weighted",
]


def main():
    runs = []
    for path in sorted(ROOT.glob("results/*/metrics.json")):
        with open(path) as f:
            runs.append(json.load(f))
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
