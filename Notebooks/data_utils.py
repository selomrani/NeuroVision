import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
DEFAULT_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, size=DEFAULT_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image corrupted")
    return cv2.resize(img, size)


def load_images(root, classes=CLASSES, size=DEFAULT_SIZE):
    images, labels, skipped = [], [], []
    for class_idx, class_name in enumerate(classes):
        folder = os.path.join(root, class_name)
        for file_name in sorted(os.listdir(folder)):
            path = os.path.join(folder, file_name)
            try:
                images.append(load_image(path, size))
                labels.append(class_idx)
            except Exception as exc:
                skipped.append((path, str(exc)))
    return np.array(images), np.array(labels), skipped


def encode_labels(labels, classes=CLASSES):
    return LabelEncoder().fit(classes).transform(labels)


def to_one_hot(labels, num_classes=len(CLASSES)):
    return tf.keras.utils.to_categorical(labels, num_classes)


def class_counts(root, classes=CLASSES):
    return {c: len(os.listdir(os.path.join(root, c))) for c in classes}


def class_balance(root, classes=CLASSES, threshold=1.5):
    counts = class_counts(root, classes)
    ratio = max(counts.values()) / min(counts.values())
    return counts, ratio, ratio > threshold


def compute_class_weights(labels, classes=CLASSES):
    labels = np.asarray(labels)
    n = len(labels)
    return {i: n / (len(classes) * int((labels == i).sum()))
            for i in range(len(classes))}


def plot_class_distribution(root, classes=CLASSES, title="Répartition des images par classe"):
    counts = class_counts(root, classes)
    plt.bar(counts.keys(), counts.values())
    plt.title(title)
    plt.show()


def list_image_files(root, classes=CLASSES):

    paths, labels = [], []
    for idx, name in enumerate(classes):
        for path in sorted(glob.glob(os.path.join(root, name, "*.jpg"))):
            paths.append(path)
            labels.append(idx)
    return np.array(paths), np.array(labels)


def decode_and_resize(path, label, size=DEFAULT_SIZE):

    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.cast(tf.image.resize(img, size), tf.uint8), label


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def make_augmentation(rotation=0.03, zoom=0.1, translation=0.1,
                      horizontal_flip=False, seed=None):

    layers = [
        tf.keras.layers.RandomRotation(rotation, fill_mode="constant", seed=seed),
        tf.keras.layers.RandomZoom(zoom, fill_mode="constant", seed=seed),
        tf.keras.layers.RandomTranslation(
            translation, translation, fill_mode="constant", seed=seed),
    ]
    if horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
    return tf.keras.Sequential(layers, name="mri_augmentation")


def split_paths(paths, labels, val_split=0.0, test_split=0.0, seed=42):

    if not 0 <= val_split + test_split < 1:
        raise ValueError("val_split + test_split must be in [0, 1)")

    holdout = val_split + test_split
    if holdout == 0:
        return {"train": (paths, labels), "val": (None, None), "test": (None, None)}

    train_x, rest_x, train_y, rest_y = train_test_split(
        paths, labels, test_size=holdout, stratify=labels, random_state=seed)

    splits = {"train": (train_x, train_y), "val": (None, None), "test": (None, None)}
    if val_split and test_split:
        rel_test = test_split / holdout
        val_x, test_x, val_y, test_y = train_test_split(
            rest_x, rest_y, test_size=rel_test, stratify=rest_y, random_state=seed)
        splits["val"], splits["test"] = (val_x, val_y), (test_x, test_y)
    elif val_split:
        splits["val"] = (rest_x, rest_y)
    else:
        splits["test"] = (rest_x, rest_y)
    return splits


def _build_dataset(paths, labels, size, batch_size, shuffle, cache, seed,
                   scale, augmentation, train):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: decode_and_resize(p, l, size),
                num_parallel_calls=AUTOTUNE)
    if scale:
        ds = ds.map(normalize, num_parallel_calls=AUTOTUNE)
    if cache is True:
        ds = ds.cache()
    elif isinstance(cache, str):
        ds = ds.cache(cache)
    if train and shuffle:
        ds = ds.shuffle(1000, seed=seed, reshuffle_each_iteration=True)
    if train and augmentation is not None:
        ds = ds.map(
            lambda x, y: (augmentation(tf.expand_dims(x, 0), training=True)[0], y),
            num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def make_datasets(root, classes=CLASSES, size=DEFAULT_SIZE, batch_size=32,
                  val_split=0.15, test_split=0.15, shuffle=True, cache=True,
                  seed=42, normalize=True, augmentation=None):
    paths, labels = list_image_files(root, classes)
    splits = split_paths(paths, labels, val_split, test_split, seed)

    datasets = {}
    for name, (split_x, split_y) in splits.items():
        datasets[name] = None if split_x is None else _build_dataset(
            split_x, split_y, size, batch_size, shuffle, cache, seed,
            scale=normalize, augmentation=augmentation, train=(name == "train"))
    return datasets, classes


def make_dataset(root, classes=CLASSES, size=DEFAULT_SIZE, batch_size=32,
                 shuffle=True, cache=True, seed=None, val_split=0.0,
                 normalize=True, augmentation=None):
    datasets, classes = make_datasets(
        root, classes=classes, size=size, batch_size=batch_size, shuffle=shuffle,
        cache=cache, seed=seed, val_split=val_split, test_split=0.0,
        normalize=normalize, augmentation=augmentation)
    return datasets["train"], datasets["val"], classes
