from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
DEFAULT_SIZE = (224, 224)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".bmp", ".png")

LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
INDEX_TO_LABEL = {index: name for name, index in LABEL_TO_INDEX.items()}
NUM_CLASSES = len(CLASSES)

AUTOTUNE = tf.data.AUTOTUNE

PREPROCESS = {
    "mobilenet": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "resnet50v2": tf.keras.applications.resnet_v2.preprocess_input,
    "vgg16": tf.keras.applications.vgg16.preprocess_input,
}


def list_image_files(root, classes=CLASSES, extensions=IMAGE_EXTENSIONS):
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
    return np.array([LABEL_TO_INDEX[name] for name in class_names], dtype=np.int64)


def one_hot(labels, num_classes=NUM_CLASSES):
    return np.eye(num_classes, dtype=np.float32)[np.asarray(labels)]


def load_image(path, size=DEFAULT_SIZE):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("corrupt or unreadable image")

    image = cv2.resize(image, size)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unexpected shape after resize: {image.shape}")

    return image


def load_images(root, classes=CLASSES, size=DEFAULT_SIZE):
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
    images, labels, failed = load_images(root, classes, size)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))

    return images[order], labels[order], failed


def class_counts(root, classes=CLASSES):
    counts = {}
    for class_name in classes:
        paths, _, _ = list_image_files(root, (class_name,))
        counts[class_name] = len(paths)
    return counts


def class_balance(root, classes=CLASSES, threshold=1.5):
    counts = class_counts(root, classes)
    smallest = min(counts.values())
    ratio = max(counts.values()) / smallest if smallest else 0.0
    return counts, ratio, ratio > threshold


def compute_class_weights(labels, classes=CLASSES):
    labels = np.asarray(labels)
    total = len(labels)
    return {
        index: total / (len(classes) * max(int((labels == index).sum()), 1))
        for index in range(len(classes))
    }


def plot_class_distribution(root, classes=CLASSES, title="Images per class"):
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
            image = load_image(paths[index])
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(class_name)
            ax.axis("off")

    fig.tight_layout()
    plt.show()


def decode_image(path, label, size=DEFAULT_SIZE):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.cast(tf.image.resize(image, size), tf.uint8)
    return image, label


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def make_augmentation(rotation=0.03, zoom=0.1, translation=0.1,
                      horizontal_flip=False, seed=None):
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
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda p, l: decode_image(p, l, size),
                          num_parallel_calls=AUTOTUNE)

    if preprocess in PREPROCESS:
        preprocess_fn = PREPROCESS[preprocess]
        dataset = dataset.map(
            lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y),
            num_parallel_calls=AUTOTUNE)
    elif preprocess == "scale01":
        dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)
    else:
        raise ValueError(f"unknown preprocess: {preprocess}")

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
