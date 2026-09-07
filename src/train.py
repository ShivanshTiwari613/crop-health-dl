"""Train one configuration and evaluate it on the held-out split.

python -m src.train --config configs/resnet50_coarse.yaml
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.data import load_datasets
from src.evaluate import evaluate
from src.models import build_model, unfreeze_top


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def fit_phase(model, data, phase, patience):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(phase["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(
        data.train,
        validation_data=data.val,
        epochs=phase["epochs"],
        class_weight=data.class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    rows = pd.DataFrame(history.history)
    rows.insert(0, "epoch", range(1, len(rows) + 1))
    rows.insert(0, "phase", phase["name"])
    return rows


def train(cfg, out_dir):
    seed_everything(cfg.get("seed", 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    data = load_datasets(cfg)
    print("train counts per class:", data.train_counts.tolist())
    print("class weights:", data.class_weights)

    model_cfg = cfg["model"]
    backbone = model_cfg["backbone"]
    model = build_model(
        backbone,
        num_classes=len(data.class_names),
        img_size=cfg["data"]["img_size"],
        dropout=model_cfg.get("dropout", 0.3),
        weights=model_cfg.get("weights", "imagenet"),
        augment=cfg["data"].get("augment", True),
    )

    patience = cfg["train"].get("early_stopping_patience", 3)
    histories = []
    start = time.time()
    for phase in cfg["train"]["phases"]:
        n = unfreeze_top(model, backbone, phase.get("unfreeze_top"))
        print(
            f"phase {phase['name']}: {phase['epochs']} epochs, lr {phase['lr']}, "
            f"{n} trainable backbone layers"
        )
        histories.append(fit_phase(model, data, phase, patience))
    train_seconds = time.time() - start

    pd.concat(histories).to_csv(out_dir / "history.csv", index=False)
    model.save(out_dir / "model.keras")

    metrics = evaluate(model, data, out_dir)
    metrics.update(
        run_name=cfg["run_name"],
        backbone=backbone,
        task=cfg["data"].get("task", "coarse"),
        img_size=cfg["data"]["img_size"],
        params=model.count_params(),
        train_seconds=round(train_seconds),
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", help="output directory, default results/<run_name>")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(args.out or f"results/{cfg['run_name']}")
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    train(cfg, out_dir)


if __name__ == "__main__":
    main()
