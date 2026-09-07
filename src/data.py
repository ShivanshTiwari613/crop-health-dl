"""PlantVillage input pipeline.

tensorflow_datasets ships PlantVillage as a single 'train' split of 54,303 leaf
images in 38 classes named "<crop>___<condition>", e.g. "Tomato___Late_blight"
or "Apple___healthy". Two label modes are supported:

  fine    the 38 original classes
  coarse  0 = healthy, 1 = diseased, read off the class name

The field data in the paper had a third "stressed" class; there is no public
equivalent, so it is not modelled here.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET = "plant_village"

# Keras application models expect their own input scaling. All three accept
# raw 0-255 floats; EfficientNet's function is the identity because the
# rescaling layer is built into the network.
PREPROCESS = {
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "inceptionv3": tf.keras.applications.inception_v3.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
}


@dataclass
class Datasets:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]
    train_counts: np.ndarray
    class_weights: dict[int, float] | None


def coarse_labels(class_names):
    return np.array([0 if "healthy" in name.lower() else 1 for name in class_names])


def balanced_class_weights(counts):
    """Weight each class by n / (k * n_c), the same rule sklearn calls 'balanced'."""
    counts = np.asarray(counts, dtype=float)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=25.0)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return tf.clip_by_value(image, 0.0, 255.0)


def count_labels(split, data_dir, label_map):
    # Skip image decoding: we only want the label column.
    ds = tfds.load(
        DATASET,
        split=split,
        data_dir=data_dir,
        decoders={"image": tfds.decode.SkipDecoding()},
    )
    labels = np.fromiter((ex["label"] for ex in tfds.as_numpy(ds)), dtype=np.int64)
    return np.bincount(label_map[labels], minlength=label_map.max() + 1)


def load_datasets(cfg):
    data_cfg = cfg["data"]
    data_dir = data_cfg.get("data_dir", "data/tfds")
    img_size = data_cfg["img_size"]
    batch_size = data_cfg["batch_size"]
    seed = cfg.get("seed", 0)
    splits = data_cfg["splits"]

    (train, val, test), info = tfds.load(
        DATASET,
        split=[splits["train"], splits["val"], splits["test"]],
        data_dir=data_dir,
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed),
    )
    fine_names = list(info.features["label"].names)

    task = data_cfg.get("task", "coarse")
    if task == "coarse":
        label_map = coarse_labels(fine_names)
        class_names = ["healthy", "diseased"]
    elif task == "fine":
        label_map = np.arange(len(fine_names))
        class_names = fine_names
    else:
        raise ValueError(f"task must be 'coarse' or 'fine', got {task!r}")

    lookup = tf.constant(label_map)
    preprocess = PREPROCESS[cfg["model"]["backbone"]]
    augment_enabled = data_cfg.get("augment", True)

    def prepare(image, label, training):
        image = tf.image.resize(tf.cast(image, tf.float32), (img_size, img_size))
        if training and augment_enabled:
            image = augment(image)
        # Augment on raw 0-255 pixels first: the backbone preprocessing maps
        # them into ranges that the augmentation ops would clip.
        return preprocess(image), tf.gather(lookup, label)

    def pipeline(ds, shuffle):
        if shuffle:
            ds = ds.shuffle(10_000, seed=seed)
        ds = ds.map(
            lambda x, y: prepare(x, y, training=shuffle), num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size)
        limit = cfg["train"].get("limit_batches")
        if limit:
            ds = ds.take(limit)
        return ds.prefetch(tf.data.AUTOTUNE)

    counts = count_labels(splits["train"], data_dir, label_map)
    weights = None
    if data_cfg.get("class_weights") == "balanced":
        weights = balanced_class_weights(counts)

    return Datasets(
        train=pipeline(train, shuffle=True),
        val=pipeline(val, shuffle=False),
        test=pipeline(test, shuffle=False),
        class_names=class_names,
        train_counts=counts,
        class_weights=weights,
    )
