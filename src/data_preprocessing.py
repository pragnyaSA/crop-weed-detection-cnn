"""
data_preprocessing.py
----------------------
Converts the Roboflow "WeedCrop" YOLOv5-format object detection dataset
(images + normalized bounding-box .txt labels) into a clean binary
image-classification dataset with two folders: `crop/` and `weed/`.

Each YOLO bounding box is cropped out of its parent image and saved as
an individual sample. This mirrors the methodology described in the
paper "Efficient Crop vs Weed Detection in Precision Agriculture: A CNN
Approach for Real-Time Decision Making" (INOCON 2024), where the model
is trained as a 2-class classifier (crop vs weed) rather than a full
object detector.

Usage
-----
    python src/data_preprocessing.py \
        --source data/raw/WeedCrop.v1i.yolov5pytorch \
        --output data/processed \
        --padding 0.08

The `source` directory is expected to have the standard Roboflow /
YOLOv5 layout:

    source/
      data.yaml
      train/images/*.jpg   train/labels/*.txt
      valid/images/*.jpg   valid/labels/*.txt
      test/images/*.jpg    test/labels/*.txt

The output directory will contain:

    output/
      train/crop/*.jpg   train/weed/*.jpg
      valid/crop/*.jpg   valid/weed/*.jpg
      test/crop/*.jpg    test/weed/*.jpg
"""

import argparse
import logging
from pathlib import Path

import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")


def load_class_names(source_dir: Path) -> dict:
    """Read class-id -> class-name mapping from the Roboflow data.yaml."""
    yaml_path = source_dir / "data.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    return {i: name for i, name in enumerate(names)}


def yolo_to_pixel_box(x_center, y_center, w, h, img_w, img_h, padding=0.0):
    """Convert normalized YOLO box (cx, cy, w, h) to pixel (x1, y1, x2, y2),
    with optional fractional padding added around the box for extra context."""
    box_w = w * img_w
    box_h = h * img_h
    cx = x_center * img_w
    cy = y_center * img_h

    pad_w = box_w * padding
    pad_h = box_h * padding

    x1 = max(0, cx - box_w / 2 - pad_w)
    y1 = max(0, cy - box_h / 2 - pad_h)
    x2 = min(img_w, cx + box_w / 2 + pad_w)
    y2 = min(img_h, cy + box_h / 2 + pad_h)
    return int(x1), int(y1), int(x2), int(y2)


def process_split(source_dir: Path, output_dir: Path, split: str,
                   class_names: dict, padding: float, min_size: int = 10):
    img_dir = source_dir / split / "images"
    label_dir = source_dir / split / "labels"

    if not img_dir.exists():
        logger.warning("Split '%s' not found at %s, skipping.", split, img_dir)
        return 0

    counts = {name: 0 for name in class_names.values()}

    for class_name in class_names.values():
        (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) +
                          list(img_dir.glob("*.jpeg")))

    for img_path in image_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Could not open %s (%s), skipping.", img_path, e)
            continue

        img_w, img_h = img.size

        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, w, h = map(float, parts[1:5])

            if class_id not in class_names:
                continue

            class_name = class_names[class_id]
            x1, y1, x2, y2 = yolo_to_pixel_box(
                x_center, y_center, w, h, img_w, img_h, padding=padding
            )

            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue

            crop = img.crop((x1, y1, x2, y2))
            out_name = f"{img_path.stem}_{i}.jpg"
            crop.save(output_dir / split / class_name / out_name, quality=95)
            counts[class_name] += 1

    total = sum(counts.values())
    logger.info("[%s] extracted %d samples -> %s", split, total, counts)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, required=True,
                         help="Path to the YOLOv5-format WeedCrop dataset root "
                              "(the folder containing data.yaml).")
    parser.add_argument("--output", type=str, default="data/processed",
                         help="Where to write the classification-ready dataset.")
    parser.add_argument("--padding", type=float, default=0.08,
                         help="Fractional padding added around each bounding box "
                              "before cropping (gives the CNN a little context).")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(source_dir)
    logger.info("Classes: %s", class_names)

    grand_total = 0
    for split in SPLITS:
        grand_total += process_split(source_dir, output_dir, split, class_names, args.padding)

    logger.info("Done. %d total classification samples written to %s", grand_total, output_dir)


if __name__ == "__main__":
    main()
