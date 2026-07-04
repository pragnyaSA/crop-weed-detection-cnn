"""
model.py
--------
CNN architecture for binary crop vs weed classification.

Design follows the methodology in the paper "Efficient Crop vs Weed
Detection in Precision Agriculture: A CNN Approach for Real-Time
Decision Making" (INOCON 2024):

  - Convolutional base: VGG16 (13 conv layers), pretrained on ImageNet,
    used as a frozen (or fine-tunable) feature extractor.
  - Input: 224 x 224 x 3 RGB images.
  - Head: Global Average Pooling -> Dense(256, ReLU) -> Dropout ->
    Dense(128, ReLU) -> Dropout -> Dense(2, Softmax)
    (2 fully-connected layers, vs. VGG16's original 3 -- this is the
    customization described in the paper).
  - Loss: categorical cross-entropy.
  - Optimizer: Adam.
"""

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16

IMG_SIZE = (224, 224)
NUM_CLASSES = 2


def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES,
                 fine_tune_at=None, learning_rate=1e-4, weights="imagenet"):
    """Build the VGG16-based crop/weed classifier.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of output classes (2: crop, weed).
    fine_tune_at : int or None
        If set, unfreezes VGG16 layers from this index onward for
        fine-tuning. If None, the whole convolutional base stays frozen
        (feature-extraction only -- fastest, good for small datasets).
    learning_rate : float
        Adam learning rate.
    weights : str or None
        Pass "imagenet" for pretrained weights (requires internet on
        first run to download), or None to train the conv base from
        scratch (useful in offline / sandboxed environments).
    """
    base_model = VGG16(include_top=False, weights=weights, input_shape=input_shape)

    if fine_tune_at is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="crop_weed_vgg16_cnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lightweight_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES,
                             learning_rate=1e-3):
    """A small from-scratch CNN with a similar block structure to VGG16
    but far fewer parameters. Useful for quick local smoke-testing,
    CPU-only training, or environments without internet access to
    download ImageNet weights. Not the architecture reported in the
    paper -- use build_model() with weights='imagenet' to reproduce
    the paper's reported accuracy.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="crop_weed_lightweight_cnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_lightweight_model()
    m.summary()
