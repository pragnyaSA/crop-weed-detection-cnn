"""
evaluate.py
-----------
Evaluates a trained crop/weed classifier on the held-out test split and
produces:
  - overall test accuracy (printed + results/test_metrics.json)
  - a classification report (precision / recall / F1 per class)
  - a confusion matrix plot (results/confusion_matrix.png)

Usage
-----
    python src/evaluate.py --model models/best_model.keras --data data/processed
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import IMG_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="models/best_model.keras")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    data_dir = Path(args.data) / args.split
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=args.batch_size,
        class_mode="categorical", shuffle=False,
    )

    class_names = list(gen.class_indices.keys())

    probs = model.predict(gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = gen.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))

    accuracy = report["accuracy"]
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix ({args.split} set) — Accuracy: {accuracy:.2%}")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    print(f"\nSaved confusion matrix to {results_dir / 'confusion_matrix.png'}")
    print(f"{args.split.capitalize()} accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
