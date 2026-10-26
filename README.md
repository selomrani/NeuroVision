# NeuroVision

Brain MRI classification into 4 classes — `glioma`, `meningioma`, `notumor`,
`pituitary` — with a CNN from scratch, then MobileNetV2, ResNet-50 and VGG-16
in transfer learning and fine-tuning. Every run is tracked in MLflow.

> Not a medical device. The output is a probability, never a diagnosis.

## Setup

```bash
uv sync
```

Data goes in `Data/raw/<class>/` (jpg/png, 224×224 resized on the fly).

## Train

```bash
uv run python -m neurovision.train                    # the 7 runs, in order
uv run python -m neurovision.train --only mobilenet-transfer
uv run python -m neurovision.train --epochs 2 --limit 3   # quick check
```

Fine-tune runs start from the matching `*-transfer` checkpoint on Azure
(`models/<arch>-transfer/best.keras`).

What gets logged for each run:

- params: architecture, preprocessing, lr, batch, epochs, total/trainable params
- metrics: loss and accuracy per epoch, validation accuracy, F1 macro and F1
  per class, training time
- artifacts: accuracy/loss curves, validation report
- the model itself, registered as `NeuroVision`

The checkpoint, curves and report are also uploaded to the `kcrowly-vast`
Azure container under `models/<run>/` and `assets/<run>/`. Without an
`AZURE_STORAGE_CONNECTION_STRING` they just stay in `assets/`.

## MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open <http://localhost:5000>, then compare the runs or run:

```bash
uv run python -m neurovision.compare
```

The `champion` alias moves to a new version only when its validation F1 is
better than the current champion. To see what is served:

```bash
uv run python -c "
from mlflow import MlflowClient
mv = MlflowClient().get_model_version_by_alias('NeuroVision', 'champion')
print('v', mv.version, 'run', mv.run_id)"
```

Load it in code with `models:/NeuroVision@champion` (the `@` is the alias
syntax; the older `models:/NeuroVision/Production` is a deprecated stage).

## App

```bash
uv run streamlit run app.py
```

The app loads whatever MLflow currently marks as `champion`, so moving the
alias changes the served model without touching the code.

## Split

Stratified 70/15/15 with `seed=42`, shared by every run, so the comparison is
fair. The test set is only used once, after training, by
`python -m neurovision.evaluate` (from the Part 2 evaluation work).
