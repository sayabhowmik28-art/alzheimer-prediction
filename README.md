# Alzheimer's Disease Prediction from MRI Scans (Deep Learning)

A complete, ready-to-run PyTorch project that classifies brain MRI images into
Alzheimer's stages using a CNN (with an optional transfer-learning backbone).

## 1. Expected dataset layout

This project expects an **image classification folder structure** (the same
layout used by the popular Kaggle "Alzheimer's Dataset" and OASIS-derived
MRI datasets):

```
data/
├── train/
│   ├── NonDemented/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
└── test/
    ├── NonDemented/
    ├── VeryMildDemented/
    ├── MildDemented/
    └── ModerateDemented/
```

You can download a matching dataset from Kaggle, e.g. search
"Alzheimer's Dataset (4 class of Images)". Unzip it so it matches the
structure above, or edit `config.py` -> `CLASS_NAMES` to match your own
classes (binary Demented/NonDemented also works — just use 2 folders).

If you already have a dataset, tell me its exact folder/column structure and
I'll adjust `src/dataset.py` to match it exactly.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Check your data before training

Before running a training job (which can take a while), validate that
`data/` is set up correctly:

```bash
python src/check_dataset.py --data_dir data
```

This catches the most common problems **before** you waste time training:
- wrong/misspelled class folder names
- empty class folders
- non-image files sitting inside a class folder
- corrupt or unreadable images
- missing `test/` classes
- strong class imbalance (warns, doesn't block — `train.py` already
  applies class-weighted loss to help with this)

It prints an image count per class for train/test, then either:
- `✓ No blocking problems found` → safe to train, or
- `✗ PROBLEMS` with a specific list → fix those first, it exits non-zero.

## 4. Train

```bash
python src/train.py --data_dir data --epochs 25 --backbone resnet18
```

Key flags (see `python src/train.py --help`):
- `--backbone` : `custom_cnn` (trained from scratch) or `resnet18` /
  `resnet50` (ImageNet transfer learning, recommended — much better accuracy
  on the small MRI datasets typically available for this task)
- `--epochs`, `--batch_size`, `--lr`
- `--freeze_backbone` : freeze pretrained conv layers, train only the
  classifier head first (faster, good for small datasets)

Training saves:
- `outputs/best_model.pt` — best checkpoint (by validation accuracy)
- `outputs/training_history.png` — loss/accuracy curves
- `outputs/training_log.csv`

## 5. Evaluate on the held-out test set

```bash
python src/evaluate.py --data_dir data --checkpoint outputs/best_model.pt
```

Produces accuracy, precision/recall/F1 per class, a confusion matrix image
(`outputs/confusion_matrix.png`), and an ROC curve for the binary case.

## 6. Predict on a single new MRI image

```bash
python src/predict.py --image path/to/scan.jpg --checkpoint outputs/best_model.pt
```

## 7. Run it as a web app (frontend + backend)

Once you have a trained checkpoint at `outputs/best_model.pt`, spin up the
full web app — a FastAPI backend that loads the model once and serves
predictions, plus a single-page frontend for uploading a scan and seeing the
per-class confidence breakdown:

```bash
uvicorn backend.app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser. Upload an MRI image,
click "Run analysis", and the readout panel fills in with the predicted
stage and a confidence bar for every class.

How the pieces connect:
- `backend/app.py` loads `outputs/best_model.pt` **once at startup** (not
  per-request) and exposes `POST /api/predict`, which accepts an uploaded
  image and returns JSON like:
  ```json
  {
    "prediction": "MildDemented",
    "confidence": 0.71,
    "all_classes": [{"label": "MildDemented", "probability": 0.71}, ...]
  }
  ```
- `frontend/index.html` is a single self-contained HTML/CSS/JS file. It
  posts the uploaded file to `/api/predict` with `fetch()` and renders the
  response — no build step, no framework, no separate frontend server.
- `app.py` also mounts `frontend/` as static files and serves `index.html`
  at `/`, so **one process serves both the API and the UI** — no CORS setup
  needed for local use (CORS is still enabled in case you split them later).

If there's no checkpoint yet, `/api/predict` returns a `503` with a clear
message instead of crashing — train a model first.

## Project structure

```
alzheimer_project/
├── config.py             # all hyperparameters & class names in one place
├── requirements.txt
├── src/
│   ├── dataset.py         # Dataset class, transforms, train/val split
│   ├── check_dataset.py    # validates data/ folder before training
│   ├── model.py            # Custom CNN + transfer-learning model builder
│   ├── train.py             # training loop, early stopping, checkpointing
│   ├── evaluate.py          # test-set metrics + confusion matrix
│   └── predict.py           # single-image inference (CLI)
├── backend/
│   └── app.py              # FastAPI server: /api/predict + serves frontend/
├── frontend/
│   └── sayan.html          # single-file upload UI, calls /api/predict
└── outputs/                # checkpoints, plots, logs land here
```

## Notes on approach

- **Why transfer learning**: public Alzheimer's MRI datasets are small
  (a few thousand images). Fine-tuning an ImageNet-pretrained ResNet
  converges faster and generalizes much better than training a CNN from
  scratch on that little data.
- **Class imbalance**: these datasets are usually imbalanced (far fewer
  "Moderate Demented" scans). `train.py` uses a class-weighted loss to
  compensate — check `outputs/training_log.csv` per-class recall to confirm
  it's actually learning the minority classes, not just predicting the
  majority class.
- **This is a portfolio/learning project, not a medical device.** Real
  clinical Alzheimer's diagnosis uses full 3D MRI volumes, PET scans,
  cognitive tests, and clinician judgment together — a 2D-slice CNN like
  this is a good way to learn the deep learning workflow, not a diagnostic
  tool. Say so explicitly if you write this up anywhere.
