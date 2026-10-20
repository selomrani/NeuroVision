"""MRI loading and preparation for NeuroVision.

Reusable part of the data discovery and preparation step: imported by the
exploration notebook and later by the training code. Functions only, no side
effects.

Covers the brief:

1. walk folders and reject non-conforming extensions
2. handle corrupt images with try-except
3. load and shuffle images and labels (same index)
4. resize with OpenCV to 224x224, 3 channels
5. keep it all here, and speed it up with tf.data
"""

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

AUTOTUNE = tf.data.AUTOTUNE


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


def decode_image(path, label, size=DEFAULT_SIZE):
    """tf.data version of load_image: decode and resize inside the graph.

    Used by make_datasets so nothing is held in memory.
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.cast(tf.image.resize(image, size), tf.uint8)
    return image, label


def normalize(image, label):
    """Scale pixels from [0, 255] to [0, 1]."""
    return tf.cast(image, tf.float32) / 255.0, label


def preprocess_mobilenet(image, label):
    """Preprocessing expected by MobileNetV2: pixels in [-1, 1].

    MobileNetV2 was trained on ImageNet at this scale. Using [0, 1] instead
    raises no error but badly degrades results with no warning.
    """
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(image, tf.float32))
    return image, label


def make_augmentation(rotation=0.03, zoom=0.1, translation=0.1,
                      horizontal_flip=False, seed=None):
    """MRI-friendly augmentation: light rotations, zoom, shifts.

    No horizontal flip by default: it swaps left and right, which is
    anatomical information.
    """
    layers = [
        tf.keras.layers.RandomRotation(rotation, fill_mode="constant", seed=seed),
        tf.keras.layers.RandomZoom(zoom, fill_mode="constant", seed=seed),
        tf.keras.layers.RandomTranslation(translation, translation,
                                          fill_mode="constant", seed=seed),
    ]
    if horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))

    return tf.keras.Sequential(layers, name="mri_augmentation")


def build_dataset(paths, labels, size=DEFAULT_SIZE, batch_size=32, shuffle=False,
                  cache=True, seed=None, augmentation=None,
                  preprocess="scale01"):
    """Build a tf.data pipeline from paths and labels.

    Preprocessing is chosen with preprocess:
      "scale01"   -> pixels in [0, 1]   (CNN from scratch)
      "mobilenet" -> pixels in [-1, 1]  (MobileNetV2)

    Augmentation only applies when passed, so only on the training set.
    """
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda p, l: decode_image(p, l, size),
                          num_parallel_calls=AUTOTUNE)

    if preprocess == "mobilenet":
        dataset = dataset.map(preprocess_mobilenet, num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)

    if cache is True:
        dataset = dataset.cache()
    elif isinstance(cache, str):
        dataset = dataset.cache(cache)

    if shuffle:
        dataset = dataset.shuffle(1000, seed=seed, reshuffle_each_iteration=True)

    if augmentation is not None:
        dataset = dataset.map(
            lambda x, y: (augmentation(tf.expand_dims(x, 0), training=True)[0], y),
            num_parallel_calls=AUTOTUNE)

    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def make_datasets(root, classes=CLASSES, size=DEFAULT_SIZE, batch_size=32,
                  val_split=0.15, test_split=0.15, seed=42, cache=True,
                  augmentation=None, preprocess="scale01"):
    """Prepare the train / validation / test sets.

    The split is stratified: each set keeps the same class proportions, and
    the seed makes it reproducible across runs. Augmentation is applied to
    training only.
    """
    paths, labels, _ = list_image_files(root, classes)
    splits = split_paths(paths, labels, val_split, test_split, seed)

    datasets = {}
    for name, (split_paths_, split_labels) in splits.items():
        if split_paths_ is None:
            datasets[name] = None
            continue
        datasets[name] = build_dataset(
            split_paths_, split_labels, size=size, batch_size=batch_size,
            shuffle=(name == "train"), cache=cache, seed=seed,
            augmentation=augmentation if name == "train" else None,
            preprocess=preprocess)

    return datasets


def split_paths(paths, labels, val_split=0.0, test_split=0.0, seed=42):
    """Split paths into train / validation / test, stratified (same class
    proportions in each set).
    """
    from sklearn.model_selection import train_test_split

    if not 0 <= val_split + test_split < 1:
        raise ValueError("val_split + test_split must be in [0, 1)")

    holdout = val_split + test_split
    if holdout == 0:
        return {"train": (paths, labels), "val": (None, None),
                "test": (None, None)}

    train_x, rest_x, train_y, rest_y = train_test_split(
        paths, labels, test_size=holdout, stratify=labels, random_state=seed)

    splits = {"train": (train_x, train_y), "val": (None, None),
              "test": (None, None)}

    if val_split and test_split:
        val_x, test_x, val_y, test_y = train_test_split(
            rest_x, rest_y, test_size=test_split / holdout,
            stratify=rest_y, random_state=seed)
        splits["val"] = (val_x, val_y)
        splits["test"] = (test_x, test_y)
    elif val_split:
        splits["val"] = (rest_x, rest_y)
    else:
        splits["test"] = (rest_x, rest_y)

    return splits
