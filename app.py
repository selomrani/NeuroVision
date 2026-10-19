"""NeuroVision — Streamlit demo for the MRI brain-tumor CNN classifier.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
DEFAULT_SIZE = (224, 224)
MODEL_DIR = Path("models")
ASSETS = Path("assets")
DATA = Path("Data/raw")

CLASS_COLORS = {
    "glioma": "#fb7185",
    "meningioma": "#fbbf24",
    "notumor": "#34d399",
    "pituitary": "#a78bfa",
}
CLASS_INFO = {
    "glioma": "Tumour arising from glial cells. Often infiltrative with irregular enhancement.",
    "meningioma": "Tumour of the meninges — usually benign and slow-growing.",
    "notumor": "No visible tumour in this slice.",
    "pituitary": "Tumour of the pituitary gland, typically at the skull base.",
}
MODEL_CHOICES = {
    "best_mri_cnn_model.keras": "Best checkpoint — lowest validation loss",
    "mri_cnn_model3.keras": "Training run 3 — 20 epochs",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp {
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(129,140,248,.16), transparent 60%),
    radial-gradient(900px 500px at 5% 5%, rgba(34,211,238,.12), transparent 55%),
    linear-gradient(165deg, #070b18 0%, #0b1224 48%, #0e1730 100%);
  background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }

h1 {
  background: linear-gradient(92deg, #22d3ee 0%, #818cf8 48%, #e879f9 100%);
  -webkit-background-clip: text; background-clip: text;
  color: transparent !important;
  font-weight: 700; letter-spacing: -.02em;
}
h2, h3 { color: #e8eefc !important; font-weight: 600 !important; }
p, li, label { color: #b9c6e3 !important; }

.hero-sub {
  color: #8fa3c8 !important; font-size: 1.02rem; font-weight: 400;
  margin-top: -.4rem;
}
.badge {
  display: inline-block; padding: .3rem .75rem; border-radius: 999px;
  font-size: .74rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; margin-right: .5rem;
  background: rgba(34,211,238,.10); color: #67e8f9;
  border: 1px solid rgba(34,211,238,.30);
}
.badge.alt { background: rgba(167,139,250,.10); color: #c4b5fd; border-color: rgba(167,139,250,.30); }

[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,.035);
  border: 1px solid rgba(255,255,255,.09) !important;
  border-radius: 16px;
  backdrop-filter: blur(8px);
}
[data-testid="stMetric"] {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 14px; padding: 16px 18px;
}
[data-testid="stMetricLabel"] { color: #8fa3c8 !important; text-transform: uppercase;
  letter-spacing: .08em; font-size: .70rem !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #f1f5fd !important; font-weight: 700 !important;
  font-size: 1.9rem !important; }

[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,.035);
  border: 1.5px dashed rgba(129,140,248,.45) !important;
  border-radius: 16px;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #22d3ee !important;
  background: rgba(34,211,238,.06);
}

.stButton > button {
  background: linear-gradient(92deg, rgba(34,211,238,.16), rgba(129,140,248,.16));
  border: 1px solid rgba(129,140,248,.42);
  color: #e8eefc !important; border-radius: 10px;
  font-weight: 600; transition: all .18s ease;
}
.stButton > button:hover {
  border-color: #22d3ee; background: linear-gradient(92deg, rgba(34,211,238,.28), rgba(129,140,248,.28));
  transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(92deg, #0891b2, #6366f1);
  border: none;
}

div[data-baseweb="select"] > div {
  background: rgba(255,255,255,.045);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 10px;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 10px; }
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  border-radius: 10px; padding: 8px 20px; color: #a9b8d8;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: linear-gradient(92deg, rgba(34,211,238,.22), rgba(129,140,248,.22));
  border-color: rgba(34,211,238,.55); color: #f1f5fd;
}

.verdict {
  border-radius: 16px; padding: 22px 24px; margin-bottom: 14px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.04);
}
.verdict-label { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
  color: #8fa3c8; font-weight: 600; }
.verdict-class { font-size: 2.15rem; font-weight: 700; line-height: 1.15; margin-top: 4px; }
.verdict-conf { font-size: .95rem; color: #a9b8d8; margin-top: 6px; }

.bar-row { display: flex; align-items: center; gap: 12px; margin: 9px 0; }
.bar-name { width: 92px; font-size: .84rem; color: #b9c6e3; font-weight: 500;
  text-transform: capitalize; }
.bar-track { flex: 1; height: 11px; border-radius: 999px;
  background: rgba(255,255,255,.07); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; transition: width .5s ease; }
.bar-val { width: 58px; text-align: right; font-size: .84rem;
  color: #e8eefc; font-variant-numeric: tabular-nums; font-weight: 600; }

.cm { border-collapse: separate; border-spacing: 5px; margin-top: 6px; }
.cm th { color: #8fa3c8; font-size: .76rem; font-weight: 600; padding: 4px 8px;
  text-transform: capitalize; }
.cm td { width: 74px; height: 58px; text-align: center; vertical-align: middle;
  border-radius: 10px; font-weight: 700; font-size: 1.02rem; color: #fff;
  border: 1px solid rgba(255,255,255,.07); }
.cm .axis { color: #6f82a8; font-size: .70rem; letter-spacing: .1em;
  text-transform: uppercase; font-weight: 600; }

.foot { color: #6f82a8; font-size: .80rem; text-align: center;
  padding: 18px 0 6px; border-top: 1px solid rgba(255,255,255,.07); margin-top: 26px; }
</style>
"""


@st.cache_resource(show_spinner="Loading model…")
def load_model(name: str) -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_DIR / name, compile=False)


@st.cache_data
def load_metrics() -> dict | None:
    path = ASSETS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def gallery(cls: str, n: int = 8, seed: int = 42) -> list[str]:
    pool = sorted(glob.glob(str(DATA / cls / "*.jpg")))
    if not pool:
        return []
    return random.Random(f"{cls}-{seed}").sample(pool, min(n, len(pool)))


def preprocess(image: Image.Image, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """RGB → model input: 224×224, float32, scaled to [0, 1]."""
    img = image.convert("RGB").resize(size, Image.BILINEAR)
    return np.expand_dims(np.asarray(img, dtype=np.float32) / 255.0, axis=0)


def predict(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    return model.predict(batch, verbose=0)[0]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def bar_row(name: str, value: float) -> str:
    color = CLASS_COLORS[name]
    return (
        f'<div class="bar-row"><div class="bar-name">{name}</div>'
        f'<div class="bar-track"><div class="bar-fill" '
        f'style="width:{value * 100:.2f}%;background:'
        f'linear-gradient(90deg,{color}bb,{color});"></div></div>'
        f'<div class="bar-val">{pct(value)}</div></div>'
    )


def confusion_matrix_html(cm: list[list[int]], classes: list[str]) -> str:
    head = "".join(f"<th>{c}</th>" for c in classes)
    rows = []
    for i, row in enumerate(cm):
        cells = []
        total = max(sum(row), 1)
        for j, v in enumerate(row):
            share = v / total
            if i == j:
                bg = f"rgba(16,185,129,{0.16 + 0.74 * share:.3f})"
                col = "#ecfdf5" if share > 0.5 else "#d1fae5"
            else:
                bg = f"rgba(244,63,94,{0.06 + 0.55 * share:.3f})"
                col = "#ffe4e6"
            cells.append(
                f'<td style="background:{bg};color:{col}">{v if v else "·"}</td>'
            )
        rows.append(
            f'<tr><th style="text-align:right">{classes[i]}</th>{"".join(cells)}</tr>'
        )
    return (
        '<table class="cm"><tr><th class="axis">actual \\ pred</th>'
        f'{head}</tr>{"".join(rows)}</table>'
    )


st.set_page_config(
    page_title="NeuroVision — MRI Tumour Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- header
st.title("🧠 NEUROVISION")
st.markdown(
    '<div class="hero-sub">Brain MRI tumour classification with a convolutional '
    "neural network — 4-class, end-to-end tf.data pipeline, deployed with Streamlit.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge">TensorFlow · Keras</span>'
    '<span class="badge alt">7,023 MRI images</span>'
    '<span class="badge">91.1% test accuracy</span>',
    unsafe_allow_html=True,
)
st.write("")

metrics = load_metrics()

# ---------------------------------------------------------------- KPI row
if metrics:
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("Test accuracy", pct(metrics["accuracy"]))
    k2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    k3.metric("Held-out test set", f"{metrics['test_size']:,}", help="Stratified 15% hold-out")
    k4.metric("Parameters", f"{metrics['params'] / 1e6:.1f}M")
else:
    st.info("Run `.venv/bin/python scripts/evaluate.py` to generate evaluation metrics.")
st.write("")

tab_demo, tab_perf, tab_info = st.tabs(
    ["🔬  Live diagnosis", "📊  Model performance", "⚗️  How it works"]
)

# ================================================================ demo
with tab_demo:
    left, right = st.columns([1, 1], gap="large")

    image: Image.Image | None = None

    with left:
        st.subheader("Input scan")
        uploaded = st.file_uploader(
            "Upload an MRI slice", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded:
            image = Image.open(uploaded)
            st.session_state.pop("sample_path", None)
            st.image(image, width="stretch")
        else:
            st.caption("…or pick a sample from the dataset")
            sample_class = st.selectbox("Sample class", list(CLASSES), key="gal_cls")
            paths = gallery(sample_class)
            if paths:
                cols = st.columns(4)
                for i, p in enumerate(paths):
                    with cols[i % 4]:
                        st.image(p, width="stretch")
                        if st.button("Use", key=f"gal{i}", width="stretch"):
                            st.session_state["sample_path"] = p
                            st.rerun()
            else:
                st.warning(f"No images under `{DATA / sample_class}`.")

            sample_path = st.session_state.get("sample_path")
            if sample_path:
                image = Image.open(sample_path)
                st.caption(f"Selected · `{Path(sample_path).name}`")

    with right:
        st.subheader("Diagnosis")
        if image is None:
            st.markdown(
                '<div class="verdict"><div class="verdict-label">Awaiting input</div>'
                '<div class="verdict-class" style="color:#55678c">Upload or select an '
                "MRI scan to classify it.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            try:
                model = load_model(
                    st.session_state.get("model_choice", list(MODEL_CHOICES)[0])
                )
            except FileNotFoundError:
                st.error(
                    f"Model file missing from `{MODEL_DIR}/`. Add the `.keras` "
                    "checkpoint and reload."
                )
                st.stop()

            probs = predict(model, preprocess(image))
            top = int(np.argmax(probs))
            label, confidence = CLASSES[top], float(probs[top])
            color = CLASS_COLORS[label]

            verdict = (
                "No tumour detected" if label == "notumor" else f"{label.capitalize()} detected"
            )
            st.markdown(
                f'<div class="verdict" style="border-color:{color}55;'
                f'background:linear-gradient(135deg,{color}1f,rgba(255,255,255,.03))">'
                f'<div class="verdict-label">Prediction</div>'
                f'<div class="verdict-class" style="color:{color}">{verdict}</div>'
                f'<div class="verdict-conf">Confidence '
                f'<strong style="color:{color}">{pct(confidence)}</strong> · '
                f"{CLASS_INFO[label]}</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown("**Class probabilities**")
            st.markdown(
                "".join(bar_row(c, float(p)) for c, p in zip(CLASSES, probs)),
                unsafe_allow_html=True,
            )

            if confidence < 0.60:
                st.warning(
                    "Low confidence — treat as screening only and review with a specialist."
                )

# ============================================================ performance
with tab_perf:
    if not metrics:
        st.info("Metrics not generated yet — run `scripts/evaluate.py`.")
    else:
        st.subheader("Confusion matrix")
        st.caption(
            f"{metrics['test_size']:,} held-out test images · rows = actual class, "
            "columns = predicted"
        )
        st.markdown(
            confusion_matrix_html(metrics["confusion_matrix"], metrics["classes"]),
            unsafe_allow_html=True,
        )
        st.write("")

        st.subheader("Per-class performance")
        cols = st.columns(len(CLASSES))
        for col, cls in zip(cols, metrics["classes"]):
            m = metrics["per_class"][cls]
            with col:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,.04);'
                    f'border:1px solid {CLASS_COLORS[cls]}44;border-radius:14px;'
                    f'padding:14px 16px;">'
                    f'<div style="color:{CLASS_COLORS[cls]};font-weight:600;'
                    f'text-transform:capitalize;font-size:.92rem;">{cls}</div>'
                    f'<div style="color:#f1f5fd;font-size:1.55rem;font-weight:700;'
                    f'margin-top:6px;">{m["f1"]:.3f}</div>'
                    f'<div style="color:#8fa3c8;font-size:.76rem;">F1 · '
                    f'P {m["precision"]:.2f} · R {m["recall"]:.2f}<br>'
                    f'n = {m["support"]}</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        st.subheader("Training curves")
        curves = ASSETS / "training_curves.png"
        if curves.exists():
            st.image(str(curves), width="stretch")
            st.caption(
                "25 epochs · Adam · sparse categorical cross-entropy · early stopping "
                "with patience 5 (best weights restored)."
            )

# ================================================================== info
with tab_info:
    st.subheader("Pipeline")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("**1 · Data**")
        st.caption(
            "7,023 MRI slices across 4 classes, resized to 224×224, corrupted files "
            "skipped automatically."
        )
    with p2:
        st.markdown("**2 · tf.data**")
        st.caption(
            "Stratified 70/15/15 split, decode → normalise → cache → shuffle → "
            "augment → batch → prefetch."
        )
    with p3:
        st.markdown("**3 · Augmentation**")
        st.caption(
            "Random rotation, zoom and translation applied to training batches only "
            "(no horizontal flip)."
        )
    with p4:
        st.markdown("**4 · Training**")
        st.caption(
            "CNN with 3 conv blocks, dropout 0.5, Adam optimiser, early stopping and "
            "best-checkpoint restore."
        )

    st.write("")
    st.subheader("Architecture")
    try:
        model = load_model(list(MODEL_CHOICES)[0])
        rows = []
        for layer in model.layers:
            try:
                shape = " → ".join(str(s) for s in layer.output.shape[1:])
            except Exception:
                shape = "—"
            rows.append((layer.name, layer.__class__.__name__, shape, layer.count_params()))
        header = (
            '<tr><th style="text-align:left">Layer</th><th style="text-align:left">'
            "Type</th><th style=\"text-align:left\">Output</th>"
            '<th style="text-align:right">Params</th></tr>'
        )
        body = "".join(
            f'<tr><td style="color:#e8eefc;padding:7px 10px">{n}</td>'
            f'<td style="color:#a9b8d8;padding:7px 10px">{t}</td>'
            f'<td style="color:#8fa3c8;padding:7px 10px;font-family:monospace">{s}</td>'
            f'<td style="color:#e8eefc;text-align:right;padding:7px 10px;'
            f'font-variant-numeric:tabular-nums">{p:,}</td></tr>'
            for n, t, s, p in rows
        )
        total = sum(r[3] for r in rows)
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;'
            f'font-size:.88rem;background:rgba(255,255,255,.03);'
            f'border:1px solid rgba(255,255,255,.09);border-radius:12px;'
            f'overflow:hidden;">{header}{body}'
            f'<tr><td colspan="3" style="color:#8fa3c8;padding:9px 10px;'
            f'font-weight:600">Total</td><td style="color:#f1f5fd;text-align:right;'
            f'padding:9px 10px;font-weight:700">{total:,}</td></tr></table>',
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.caption("Model checkpoint unavailable — architecture table skipped.")

    st.write("")
    st.subheader("Classes")
    for cls in CLASSES:
        st.markdown(
            f'<span style="color:{CLASS_COLORS[cls]};font-weight:600;'
            f'text-transform:capitalize">● {cls}</span> — {CLASS_INFO[cls]}'
        )

st.markdown(
    '<div class="foot">NeuroVision · demonstration project — not a medical device. '
    "Outputs must never be used as a clinical diagnosis.</div>",
    unsafe_allow_html=True,
)
