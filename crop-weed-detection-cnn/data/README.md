# Data

This project uses the **WeedCrop** dataset (Roboflow, YOLOv5 PyTorch export format),
2,822 field images labeled with bounding boxes for two classes: `crop` and `weed`.

## 1. Download

Get the dataset from Kaggle / Roboflow and place it under `data/raw/` so that it looks like:

```
data/raw/WeedCrop.v1i.yolov5pytorch/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Source: search "WeedCrop" on Roboflow Universe or Kaggle, YOLOv5 PyTorch TXT export.

## 2. Convert to a classification dataset

The original dataset is in YOLO object-detection format (bounding boxes). This
project trains a **binary image classifier** (crop vs weed), matching the
methodology in the paper. Run:

```bash
python src/data_preprocessing.py \
    --source data/raw/WeedCrop.v1i.yolov5pytorch \
    --output data/processed
```

This crops every labeled bounding box out of its source image and sorts the
crops into `data/processed/{train,valid,test}/{crop,weed}/`, ready for
`ImageDataGenerator.flow_from_directory`.

## Quick-start sample

`data/sample/` in this repo is a small, already-preprocessed subset (60
train / 20 valid / 20 test images per class) checked directly into git so
you can run `evaluate.py`, `predict.py`, and the Streamlit app immediately
against the included `models/best_model.keras` demo checkpoint, without
downloading anything.

## Class imbalance note

The raw dataset is naturally imbalanced (far more `weed`-labeled boxes than
`crop`-labeled boxes). If you train on the full dataset, consider passing
`class_weight` to `model.fit` or oversampling the minority class — the
`train.py` script is a good place to add this.
