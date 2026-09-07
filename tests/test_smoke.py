"""Fast checks that need no dataset: label mapping, weights, and one training step per backbone."""

import numpy as np
import pytest
import tensorflow as tf

from src.data import PREPROCESS, augment, balanced_class_weights, coarse_labels
from src.models import BACKBONES, build_model, unfreeze_top

IMG_SIZE = 96  # InceptionV3 refuses inputs smaller than 75


def test_coarse_labels_from_class_names():
    names = ["Apple___Apple_scab", "Apple___healthy", "Tomato___Late_blight", "Corn___HEALTHY"]
    assert coarse_labels(names).tolist() == [1, 0, 1, 0]


def test_balanced_class_weights_matches_sklearn_rule():
    weights = balanced_class_weights([10, 30])
    assert weights == {0: 2.0, 1: 40 / 60}


def test_augment_keeps_pixel_range_and_shape():
    image = tf.random.uniform((IMG_SIZE, IMG_SIZE, 3), 0, 255)
    out = augment(image)
    assert out.shape == image.shape
    assert float(tf.reduce_min(out)) >= 0 and float(tf.reduce_max(out)) <= 255


def test_every_backbone_has_a_preprocess_function():
    assert set(PREPROCESS) == set(BACKBONES)


@pytest.mark.parametrize("backbone", sorted(BACKBONES))
def test_backbone_builds_and_trains_one_step(backbone):
    model = build_model(backbone, num_classes=3, img_size=IMG_SIZE, weights=None)
    assert model.output_shape == (None, 3)

    x = np.random.rand(4, IMG_SIZE, IMG_SIZE, 3).astype("float32") * 255
    y = np.array([0, 1, 2, 1])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    loss = float(model.train_on_batch(x, y))
    assert np.isfinite(loss)


@pytest.mark.parametrize("backbone", sorted(BACKBONES))
def test_unfreeze_top_skips_batchnorm(backbone):
    model = build_model(backbone, num_classes=3, img_size=IMG_SIZE, weights=None)
    n = unfreeze_top(model, backbone, 10)
    assert 0 < n <= 10
    base = model.get_layer(backbone)
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert not layer.trainable
    assert unfreeze_top(model, backbone, None) == 0
