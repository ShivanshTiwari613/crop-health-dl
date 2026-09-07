"""ImageNet-pretrained backbones with a small classification head."""

import tensorflow as tf

BACKBONES = {
    "resnet50": tf.keras.applications.ResNet50,
    "inceptionv3": tf.keras.applications.InceptionV3,
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
}


def augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augment",
    )


def build_model(backbone, num_classes, img_size, dropout=0.3, weights="imagenet", augment=True):
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {backbone!r}")

    base = BACKBONES[backbone](
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
        name=backbone,
    )
    base.trainable = False

    inputs = tf.keras.Input((img_size, img_size, 3), name="image")
    x = augmentation()(inputs) if augment else inputs
    # training=False keeps BatchNorm in inference mode even after layers are
    # unfrozen; updating BN statistics on small batches destroys the
    # pretrained features.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=backbone)


def unfreeze_top(model, backbone, n_layers):
    """Make the last n_layers of the backbone trainable, BatchNorm excluded.

    n_layers=None or 0 freezes the whole backbone. Returns the number of
    layers that ended up trainable.
    """
    base = model.get_layer(backbone)
    if not n_layers:
        base.trainable = False
        return 0

    base.trainable = True
    first = len(base.layers) - n_layers
    trainable = 0
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= first and not isinstance(layer, tf.keras.layers.BatchNormalization)
        trainable += layer.trainable
    return trainable
