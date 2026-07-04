"""
train.py
--------
Trains the crop vs weed CNN classifier and saves:
  - the best model checkpoint (models/best_model.keras)
  - training curves (results/accuracy_plot.png, results/loss_plot.png)
  - a training history CSV (results/training_history.csv)

Usage
-----
    python src/train.py --data data/processed --epochs 25 --batch-size 32

    # Quick CPU smoke test with the lightweight architecture and no
    # pretrained weights (no internet required):
    python src/train.py --data data/processed --epochs 3 --lightweight
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import IMG_SIZE, build_lightweight_model, build_model


def compute_class_weights(train_gen):
    """Balance the loss so the rare class (typically 'crop') isn't ignored."""
    counts = Counter(train_gen.classes)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
    return weights


def get_generators(data_dir: Path, batch_size: int, img_size=IMG_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        data_dir / "train", target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=True,
    )
    val_gen = eval_datagen.flow_from_directory(
        data_dir / "valid", target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen


def plot_history(history, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Epoch vs Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Epoch vs Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_plot.png", dpi=150)
    plt.close()

    with open(results_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(history.history.keys())
        writer.writerow(["epoch"] + keys)
        for i in range(len(history.history[keys[0]])):
            writer.writerow([i + 1] + [history.history[k][i] for k in keys])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/processed",
                         help="Path to processed classification dataset "
                              "(must contain train/ and valid/ subfolders).")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune-at", type=int, default=None,
                         help="Unfreeze VGG16 layers from this index onward.")
    parser.add_argument("--lightweight", action="store_true",
                         help="Use the small from-scratch CNN instead of "
                              "VGG16 transfer learning (no internet needed).")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                         help="Optional cap on steps/epoch (useful for quick smoke tests).")
    parser.add_argument("--validation-steps", type=int, default=None,
                         help="Optional cap on validation steps (useful for quick smoke tests).")
    parser.add_argument("--no-class-weight", action="store_true",
                         help="Disable automatic class balancing (on by default, since "
                              "this dataset has far more 'weed' than 'crop' samples).")
    args = parser.parse_args()

    data_dir = Path(args.data)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen = get_generators(data_dir, args.batch_size)
    print("Class indices:", train_gen.class_indices)

    class_weight = None
    if not args.no_class_weight:
        class_weight = compute_class_weights(train_gen)
        print("Class weights (balancing rare class):", class_weight)

    if args.lightweight:
        model = build_lightweight_model(learning_rate=args.lr)
    else:
        model = build_model(fine_tune_at=args.fine_tune_at, learning_rate=args.lr)

    model.summary()

    callbacks = [
        ModelCheckpoint(str(models_dir / "best_model.keras"),
                         monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=6,
                       restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        class_weight=class_weight,
    )

    plot_history(history, results_dir)
    model.save(models_dir / "final_model.keras")
    print(f"Training complete. Best model saved to {models_dir / 'best_model.keras'}")


if __name__ == "__main__":
    main()
