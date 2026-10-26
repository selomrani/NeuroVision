"""Train the seven runs and track them in MLflow.

    python -m neurovision.train                       # all seven
    python -m neurovision.train --only vgg16-transfer
    python -m neurovision.train --epochs 2 --limit 3  # quick local check
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score

from neurovision.data import CLASSES, make_augmentation, make_datasets
from neurovision.models import build_scratch, build_transfer, unfreeze

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
MODEL = "NeuroVision"
ALIAS = "champion"
CONTAINER = "kcrowly-vast"

# name, architecture, preprocessing, batch, lr, epochs
EXPERIMENTS = [
    ("cnn-scratch", "scratch", "scale01", 32, 1e-3, 30),
    ("mobilenet-transfer", "mobilenet", "mobilenet", 32, 1e-3, 20),
    ("mobilenet-finetune", "mobilenet", "mobilenet", 32, 1e-4, 15),
    ("resnet50-transfer", "resnet50", "resnet50", 32, 1e-3, 20),
    ("resnet50-finetune", "resnet50", "resnet50", 32, 1e-5, 15),
    ("vgg16-transfer", "vgg16", "vgg16", 16, 1e-3, 20),
    ("vgg16-finetune", "vgg16", "vgg16", 16, 1e-5, 15),
]

load_dotenv()


def container():
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        return None
    return BlobServiceClient.from_connection_string(conn).get_container_client(CONTAINER)


def upload(path, blob):
    client = container()
    if client is None:
        print(f"no Azure key, {path.name} stays local")
        return
    with open(path, "rb") as f:
        client.upload_blob(blob, f, overwrite=True)
    print(f"uploaded {blob}")


def download(blob, path):
    client = container()
    if client is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            client.download_blob(blob).readinto(f)
        return True
    except Exception:
        return False


def load_transfer(arch, lr):
    # fine-tune needs the frozen-base weights from the transfer run
    path = ASSETS / f"{arch}-transfer.keras"
    if not download(f"models/{arch}-transfer/best.keras", path) and not path.exists():
        raise SystemExit(f"run {arch}-transfer first")
    return unfreeze(tf.keras.models.load_model(path, compile=False), lr=lr)


def predict_all(model, ds):
    y_true, y_pred = [], []
    for x, y in ds:
        y_pred.append(model.predict(x, verbose=0).argmax(axis=1))
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def log_history(history):
    for step, (loss, acc, vloss, vacc) in enumerate(zip(
            history.history["loss"], history.history["accuracy"],
            history.history["val_loss"], history.history["val_accuracy"])):
        mlflow.log_metrics({"train_loss": loss, "train_accuracy": acc,
                            "val_loss": vloss, "val_accuracy": vacc}, step=step)


def save_curves(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, key, title in ((ax1, "accuracy", "accuracy"), (ax2, "loss", "loss")):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history[f"val_{key}"], label="val")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def promote(version, f1):
    client = mlflow.MlflowClient()
    try:
        current = client.get_model_version_by_alias(MODEL, ALIAS)
        best = client.get_run(current.run_id).data.metrics.get("val_f1_macro", 0)
    except Exception:
        current, best = None, -1.0

    if f1 > best:
        client.set_registered_model_alias(MODEL, ALIAS, version)
        print(f"champion -> v{version} (f1 {f1:.3f})")
    else:
        print(f"champion stays v{current.version}, f1 {best:.3f} >= {f1:.3f}")


def train_one(name, arch, preprocess, batch, lr, epochs, limit=None):
    tf.keras.backend.clear_session()
    sets = make_datasets(
        str(ROOT / "Data" / "raw"),
        batch_size=batch,
        preprocess=preprocess,
        augmentation=make_augmentation(0.03, 0.1, 0.1),
    )
    train, val = sets["train"], sets["val"]
    if limit:
        train, val = train.take(limit), val.take(limit)

    if arch == "scratch":
        model = build_scratch(lr)
    elif name.endswith("finetune"):
        model = load_transfer(arch, lr)
    else:
        model = build_transfer(arch, lr)

    checkpoint = ASSETS / f"{name}.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.BackupAndRestore(str(ROOT / "backups" / name)),
    ]

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "architecture": arch,
            "preprocess": preprocess,
            "batch_size": batch,
            "learning_rate": lr,
            "epochs": epochs,
            "total_params": int(model.count_params()),
            "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        })

        started = time.perf_counter()
        history = model.fit(train, validation_data=val, epochs=epochs,
                            callbacks=callbacks, verbose=2)
        minutes = (time.perf_counter() - started) / 60
        log_history(history)

        y_true, y_pred = predict_all(model, val)
        metrics = {
            "val_accuracy": float((y_true == y_pred).mean()),
            "val_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "duration_minutes": minutes,
        }
        for index, class_name in enumerate(CLASSES):
            metrics[f"val_f1_{class_name}"] = float(
                f1_score(y_true, y_pred, labels=[index], average=None, zero_division=0)[0])
        mlflow.log_metrics(metrics)

        curves = ASSETS / f"{name}-curves.png"
        report = ASSETS / f"{name}-report.txt"
        save_curves(history, curves)
        report.write_text(classification_report(
            y_true, y_pred, target_names=list(CLASSES), zero_division=0))

        mlflow.log_artifact(str(curves), artifact_path="report")
        mlflow.log_artifact(str(report), artifact_path="report")
        upload(checkpoint, f"models/{name}/best.keras")
        upload(curves, f"assets/{name}/curves.png")
        upload(report, f"assets/{name}/report.txt")

        info = mlflow.tensorflow.log_model(model, name="model")
        version = mlflow.register_model(info.model_uri, MODEL).version
        mlflow.set_tag("model_version", version)
        print(f"{name}: val f1 {metrics['val_f1_macro']:.3f} in {minutes:.1f} min")
        promote(version, metrics["val_f1_macro"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run a single experiment by name")
    parser.add_argument("--epochs", type=int, help="override the epoch count")
    parser.add_argument("--limit", type=int, help="use only N batches, for a quick test")
    args = parser.parse_args()

    ASSETS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("NeuroVision")

    for name, arch, preprocess, batch, lr, epochs in EXPERIMENTS:
        if args.only and args.only != name:
            continue
        train_one(name, arch, preprocess, batch, lr,
                  args.epochs or epochs, args.limit)


if __name__ == "__main__":
    main()
