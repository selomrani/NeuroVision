import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
DEFAULT_SIZE = (224, 224)


def load_image(path, size=DEFAULT_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image corrupted")
    return cv2.resize(img, size)


def load_images(root, classes=CLASSES, size=DEFAULT_SIZE):
    images, labels, skipped = [], [], []
    for class_name in classes:
        folder = os.path.join(root, class_name)
        for file_name in sorted(os.listdir(folder)):
            path = os.path.join(folder, file_name)
            try:
                images.append(load_image(path, size))
                labels.append(class_name)
            except Exception as exc:
                skipped.append((path, str(exc)))
    return np.array(images), np.array(labels), skipped


def plot_class_distribution(root, classes=CLASSES, title="Répartition des images par classe"):
    counts = {c: len(os.listdir(os.path.join(root, c))) for c in classes}
    plt.bar(counts.keys(), counts.values())
    plt.title(title)
    plt.show()
