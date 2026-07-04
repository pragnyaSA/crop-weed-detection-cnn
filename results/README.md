# Results

This folder is populated by `src/train.py` and `src/evaluate.py`:

- `accuracy_plot.png` / `loss_plot.png` — training vs validation curves
- `training_history.csv` — per-epoch metrics
- `confusion_matrix.png` — test-set confusion matrix
- `test_metrics.json` — precision / recall / F1 per class

The plots currently checked in were produced by a **short pipeline
verification run** on `data/sample/` (a small tracked subset of ~4,900
cropped images, trained with the lightweight from-scratch CNN in
`src/model.py` for a few epochs so it runs without internet access to
download ImageNet weights) — they confirm the full pipeline works
end-to-end, not the paper's reported accuracy. To reproduce numbers close to
the paper's reported 94.5% validation / 95% test accuracy, train the full
VGG16 transfer-learning model (`build_model` in `src/model.py`) on the
complete dataset (`data/processed/`, generated from the full raw dataset via
`src/data_preprocessing.py`) for ~25 epochs, e.g.:

```bash
python src/train.py --data data/processed --epochs 25 --batch-size 32
```
