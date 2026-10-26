"""Evaluate the trained model on the held-out test split.

Writes:
  assets/metrics.json        - overall + per-class metrics, confusion matrix
  assets/training_curves.png - accuracy/loss curves extracted from CNN.ipynb

Run:  .venv/bin/python scripts/evaluate.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))
from data_utils import CLASSES, list_image_files, split_paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data" / "raw"
MODEL = ROOT / "models" / "best_mri_cnn_model.keras"
ASSETS = ROOT / "assets"
SIZE = (224, 224)
BATCH = 32


def extract_curves() -> bool:
    """Pull the training-curves figure out of CNN.ipynb cell outputs."""
    import json as _json

    nb = _json.loads((ROOT / "Notebooks" / "CNN.ipynb").read_text())
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            png = (out.get("data") or {}).get("image/png")
            if png:
                (ASSETS / "training_curves.png").write_bytes(
                    base64.b64decode(png if isinstance(png, str) else "".join(png))
                )
                return True
    return False


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    paths, labels = list_image_files(str(DATA), CLASSES)
    splits = split_paths(paths, labels, val_split=0.15, test_split=0.15, seed=42)
    test_x, test_y = splits["test"]
    print(f"test images: {len(test_x)}")

    model = tf.keras.models.load_model(MODEL, compile=False)

    def load(p):
        raw = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        return tf.cast(tf.image.resize(raw, SIZE), tf.float32) / 255.0

    ds = tf.data.Dataset.from_tensor_slices(test_x).map(load, tf.data.AUTOTUNE).batch(BATCH)
    probs = model.predict(ds, verbose=1)
    preds = probs.argmax(axis=1)

    cm = confusion_matrix(test_y, preds, labels=list(range(len(CLASSES))))
    report = classification_report(
        test_y, preds, labels=list(range(len(CLASSES))),
        target_names=list(CLASSES), output_dict=True, zero_division=0,
    )

    params = int(model.count_params())
    metrics = {
        "model": MODEL.name,
        "test_size": int(len(test_x)),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "params": params,
        "trainable_params": int(sum(
            np.prod(v.shape) for v in model.trainable_weights
        )),
        "classes": list(CLASSES),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }
            for c in CLASSES
        },
    }
    (ASSETS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"accuracy: {metrics['accuracy']:.4f}  macro F1: {metrics['macro_f1']:.4f}")
    print(f"-> {ASSETS / 'metrics.json'}")

    if extract_curves():
        print(f"-> {ASSETS / 'training_curves.png'}")
    else:
        print("!! no figure found in CNN.ipynb")


if __name__ == "__main__":
    main()
