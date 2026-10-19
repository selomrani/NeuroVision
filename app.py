"""NeuroVision — Streamlit demo for the MRI brain-tumor CNN classifier.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
DEFAULT_SIZE = (224, 224)
MODEL_DIR = "models"

CLASS_INFO = {
    "glioma": "Glioma — a tumor arising from glial cells. Often infiltrative; imaging shows irregular enhancement.",
    "meningioma": "Meningioma — a tumor of the meninges, usually benign and slow-growing.",
    "notumor": "No tumor — the scan shows no visible tumor in this slice.",
    "pituitary": "Pituitary — a tumor of the pituitary gland, typically located at the skull base.",
}

MODEL_CHOICES = {
    "best_mri_cnn_model.keras": "Best model (checkpoint, lowest validation loss)",
    "mri_cnn_model3.keras": "Training run 3 (20 epochs)",
}


@st.cache_resource(show_spinner="Loading model…")
def load_model(name: str) -> tf.keras.Model:
    """Load and cache a Keras model; weights only are needed for inference."""
    return tf.keras.models.load_model(f"{MODEL_DIR}/{name}", compile=False)


def preprocess(image: Image.Image, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """RGB → model input: resize to 224×224, float32, scaled to [0, 1]."""
    img = image.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    return model.predict(batch, verbose=0)[0]


st.set_page_config(page_title="NeuroVision", page_icon="🧠", layout="wide")

st.title("🧠 NeuroVision — MRI Tumor Classification")
st.caption(
    "Upload a brain MRI slice to classify it as glioma, meningioma, no tumor, or pituitary."
)

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Checkpoint", list(MODEL_CHOICES))
    st.caption(MODEL_CHOICES[model_name])
    st.divider()
    st.markdown(
        "**Pipeline**\n"
        "- Input: 224 × 224 × 3 RGB\n"
        "- Normalisation: pixels ÷ 255 → [0, 1]\n"
        f"- Classes: {', '.join(CLASSES)}"
    )
    st.divider()
    st.markdown(
        "*Not a medical device — for demonstration and education only.*"
    )

try:
    model = load_model(model_name)
except FileNotFoundError:
    st.error(
        f"Model file `models/{model_name}` not found. Place the `.keras` "
        "checkpoint in the `models/` directory and reload."
    )
    st.stop()

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Upload an image to begin, or pick a sample below.")
    sample_class = st.selectbox("Or load a sample from the dataset", CLASSES)
    if st.button("Load sample"):
        st.session_state["sample_class"] = sample_class

    sample_class = st.session_state.get("sample_class")
    if sample_class:
        import glob
        import random

        candidates = sorted(glob.glob(f"Data/raw/{sample_class}/*.jpg"))
        if candidates:
            path = random.choice(candidates)
            st.session_state["image"] = Image.open(path)
            st.caption(f"Sample: `{path}`")
        else:
            st.warning(f"No sample images found under `Data/raw/{sample_class}/`.")
    if "image" not in st.session_state:
        st.stop()
    image = st.session_state["image"]
else:
    image = Image.open(uploaded)
    st.session_state["image"] = image

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    st.image(image, width="stretch")

with right:
    st.subheader("Prediction")
    probs = predict(model, preprocess(image))
    top = int(np.argmax(probs))
    label = CLASSES[top]
    confidence = float(probs[top])

    if label == "notumor":
        st.success(f"**No tumor detected** — {confidence:.1%} confidence")
    else:
        st.error(f"**{label.capitalize()} detected** — {confidence:.1%} confidence")

    st.markdown(CLASS_INFO[label])

    st.markdown("#### Class probabilities")
    for cls, prob in sorted(zip(CLASSES, probs), key=lambda kv: -kv[1]):
        st.progress(float(prob), text=f"{cls}: {prob:.1%}")

    if confidence < 0.6:
        st.warning("Low confidence — consider reviewing the scan with a specialist.")
