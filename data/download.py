"""Download and prepare PlantVillage under data/tfds (about 0.8 GB)."""

import tensorflow_datasets as tfds

builder = tfds.builder("plant_village", data_dir="data/tfds")
builder.download_and_prepare()

info = builder.info
print(f"{info.splits['train'].num_examples} images, {info.features['label'].num_classes} classes")
