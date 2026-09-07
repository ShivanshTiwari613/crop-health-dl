"""Test-split metrics, confusion matrix and per-class report.

Runs automatically at the end of training; can also be re-run on a saved model:

    python -m src.evaluate --run results/resnet50_coarse
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import load_datasets

plt.switch_backend("Agg")


def predict(model, dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model.predict_on_batch(images)
        y_true.append(labels.numpy())
        y_pred.append(probs.argmax(axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion_matrix(cm, class_names, path):
    n = len(class_names)
    size = max(4, 0.35 * n)
    fontsize = 6 if n > 10 else 9
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), class_names, rotation=90, fontsize=fontsize)
    ax.set_yticks(range(n), class_names, fontsize=fontsize)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if n <= 10:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(model, data, out_dir):
    y_true, y_pred = predict(model, data.test)
    labels = list(range(len(data.class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, data.class_names, out_dir / "confusion_matrix.png")

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=data.class_names, zero_division=0
    )
    (out_dir / "classification_report.txt").write_text(report)

    def score(fn, **kw):
        return round(float(fn(y_true, y_pred, **kw)), 4)

    return {
        "n_test": int(len(y_true)),
        "accuracy": score(accuracy_score),
        "f1_macro": score(f1_score, average="macro", zero_division=0),
        "f1_weighted": score(f1_score, average="weighted", zero_division=0),
        "precision_macro": score(precision_score, average="macro", zero_division=0),
        "recall_macro": score(recall_score, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="a results/<run_name> directory")
    args = parser.parse_args()
    run_dir = Path(args.run)

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    data = load_datasets(cfg)
    model = tf.keras.models.load_model(run_dir / "model.keras")

    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    metrics.update(evaluate(model, data, run_dir))
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
