"""
predict.py
----------
Real-time / single-image inference for the crop vs weed classifier.

Usage
-----
    # Single image
    python src/predict.py --model models/best_model.keras --image path/to/img.jpg

    # Every image in a folder
    python src/predict.py --model models/best_model.keras --folder path/to/images/

    # Real-time webcam demo (requires a connected camera)
    python src/predict.py --model models/best_model.keras --webcam
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from model import IMG_SIZE

CLASS_NAMES = ["crop", "weed"]


def preprocess(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(model, image_path):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Could not read {image_path}")
        return
    batch = preprocess(frame)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    print(f"{image_path.name}: {CLASS_NAMES[idx]}  (confidence: {probs[idx]:.2%})")


def run_webcam(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch = preprocess(frame)
        probs = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = f"{CLASS_NAMES[idx]} ({probs[idx]:.1%})"
        color = (0, 200, 0) if CLASS_NAMES[idx] == "crop" else (0, 0, 255)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Crop vs Weed - Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="models/best_model.keras")
    parser.add_argument("--image", type=str, help="Path to a single image.")
    parser.add_argument("--folder", type=str, help="Path to a folder of images.")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam inference.")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.webcam:
        run_webcam(model)
    elif args.image:
        predict_image(model, Path(args.image))
    elif args.folder:
        folder = Path(args.folder)
        for img_path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
            predict_image(model, img_path)
    else:
        parser.error("Provide one of --image, --folder, or --webcam.")


if __name__ == "__main__":
    main()
