"""MRI loading and preparation for NeuroVision.

Reusable part of the data discovery and preparation step: imported by the
exploration notebook and later by the training code. Functions only, no side
effects.

Covers the brief:

1. walk folders and reject non-conforming extensions
2. handle corrupt images with try-except
3. load and shuffle images and labels (same index)
4. resize with OpenCV to 224x224, 3 channels
"""

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# One folder per class; the order fixes the label encoding:
# glioma=0, meningioma=1, notumor=2, pituitary=3
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")

DEFAULT_SIZE = (224, 224)

# Extensions allowed by the brief. Everything else is rejected.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".bmp", ".png")

# Label encoding: class name -> integer index. The order of CLASSES fixes it.
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
INDEX_TO_LABEL = {index: name for name, index in LABEL_TO_INDEX.items()}
NUM_CLASSES = len(CLASSES)


def list_image_files(root, classes=CLASSES, extensions=IMAGE_EXTENSIONS):
    """Walk the folders and keep only images with an allowed extension.

    Returns the accepted paths, their label (the class index) and the rejected
    files with a reason.
    """
    paths, labels, rejected = [], [], []

    for class_index, class_name in enumerate(classes):
        folder = os.path.join(root, class_name)
        if not os.path.isdir(folder):
            continue

        for file_name in sorted(os.listdir(folder)):
            path = os.path.join(folder, file_name)
            if not os.path.isfile(path):
                continue

            extension = os.path.splitext(file_name)[1].lower()
            if extension not in extensions:
                rejected.append((path, f"unsupported extension: {extension}"))
                continue

            paths.append(path)
            labels.append(class_index)

    return np.array(paths), np.array(labels), rejected


def encode_labels(class_names):
    """Map class names to integer labels 0..N-1.

    Integer labels are what sparse categorical cross-entropy expects.
    """
    return np.array([LABEL_TO_INDEX[name] for name in class_names], dtype=np.int64)


def one_hot(labels, num_classes=NUM_CLASSES):
    """One-hot encode integer labels, for categorical cross-entropy."""
    return np.eye(num_classes, dtype=np.float32)[np.asarray(labels)]


def load_image(path, size=DEFAULT_SIZE):
    """Read an image with OpenCV and resize to 224x224 with 3 channels.

    cv2.IMREAD_COLOR forces 3 channels even for a greyscale file, otherwise a
    black and white MRI comes out (224, 224) instead of (224, 224, 3) and the
    model cannot read it.

    Raises ValueError on an unreadable file; the caller decides what to do.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("corrupt or unreadable image")

    image = cv2.resize(image, size)

    # Safety check: after resize, confirm the 3 channels are there.
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unexpected shape after resize: {image.shape}")

    return image


def load_images(root, classes=CLASSES, size=DEFAULT_SIZE):
    """Load every image of the dataset into memory.

    Corrupt files are skipped and counted instead of stopping the walk: one
    unreadable MRI must not fail the whole load.

    Returns images (N, 224, 224, 3), labels (N,) and the failed files.
    """
    paths, labels, _ = list_image_files(root, classes)

    images, kept_labels, failed = [], [], []
    for path, label in zip(paths, labels):
        try:
            images.append(load_image(path, size))
            kept_labels.append(label)
        except Exception as error:
            failed.append((path, str(error)))

    return np.array(images), np.array(kept_labels), failed


def load_and_shuffle(root, classes=CLASSES, size=DEFAULT_SIZE, seed=42):
    """Load the images, then shuffle images and labels together.

    One permutation is applied to both arrays: shuffling them separately would
    break the image/label pairing and train the model on noise.

    Returns shuffled images, shuffled labels and the errors.
    """
    images, labels, failed = load_images(root, classes, size)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))

    return images[order], labels[order], failed


def class_counts(root, classes=CLASSES):
    """Count images per class, keeping only allowed extensions."""
    counts = {}
    for class_name in classes:
        paths, _, _ = list_image_files(root, (class_name,))
        counts[class_name] = len(paths)
    return counts


def class_balance(root, classes=CLASSES, threshold=1.5):
    """Return the counts, the largest/smallest class ratio, and whether that
    ratio exceeds the threshold (so whether rebalancing is needed).
    """
    counts = class_counts(root, classes)
    smallest = min(counts.values())
    ratio = max(counts.values()) / smallest if smallest else 0.0
    return counts, ratio, ratio > threshold


def compute_class_weights(labels, classes=CLASSES):
    """Weight per class to compensate an imbalance.

    A small class gets a larger weight, so the model does not ignore it in
    favour of the majority classes.
    """
    labels = np.asarray(labels)
    total = len(labels)
    return {
        index: total / (len(classes) * max(int((labels == index).sum()), 1))
        for index in range(len(classes))
    }


def plot_class_distribution(root, classes=CLASSES,
                            title="Images per class"):
    """Plot a histogram of the number of images per class."""
    counts = class_counts(root, classes)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title(title)
    ax.set_ylabel("images")
    ax.bar_label(bars)
    fig.tight_layout()
    plt.show()

    return counts


def show_examples(root, classes=CLASSES, per_class=3, seed=42):
    """Plot a grid of examples, one row per class."""
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(len(classes), per_class,
                             figsize=(3 * per_class, 3 * len(classes)))
    axes = np.atleast_2d(axes)

    for row, class_name in enumerate(classes):
        paths, _, _ = list_image_files(root, (class_name,))
        chosen = rng.choice(len(paths), size=min(per_class, len(paths)),
                            replace=False)

        for col, index in enumerate(chosen):
            ax = axes[row, col]
            # Images are BGR (OpenCV order): convert to RGB so colours show right.
            image = load_image(paths[index])
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(class_name)
            ax.axis("off")

    fig.tight_layout()
    plt.show()
