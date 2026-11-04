import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "module_2" / "models"

st.set_page_config(
    page_title="Breast Tissue Classification",
    page_icon="🔬"
)

st.title("Breast Tissue Image Classification")
st.write("Upload a microscopic breast-tissue image.")

st.warning(
    "Educational research demonstration. This model has limited "
    "performance on unseen patients and must not be used for diagnosis."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        str(MODEL_DIR / "breast_tissue_classifier.keras"),
        compile=False
    )

@st.cache_data
def load_config():
    with open(MODEL_DIR / "model_config.json", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
config = load_config()
threshold = float(config["threshold"])

uploaded_file = st.file_uploader(
    "Choose a tissue image",
    type=["png", "jpg", "jpeg"]
)

st.caption(
    "Designed for BreaKHis-style microscopic tissue images "
    "at 40×, 100×, 200×, or 400× magnification."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (OSError, ValueError):
        st.error("Unable to read this image. Please upload a valid image file.")
        st.stop()

    st.image(image, caption="Uploaded image", width=450)

    if st.button("Classify image", type="primary"):
        with st.spinner("Analyzing image..."):
            height, width = config["image_size"]

            # Match the Colab resizing and pixel range.
            image_array = np.asarray(image, dtype=np.float32)
            resized = tf.image.resize(
                image_array,
                [height, width],
                method="bilinear"
            )
            batch = tf.expand_dims(resized, axis=0)

            # Scaling is already included inside the saved model.
            score = float(model(batch, training=False).numpy()[0, 0])

        predicted_class = (
            "Malignant" if score >= threshold else "Benign"
        )

        st.subheader(f"Model prediction: {predicted_class}")
        st.metric("Malignant model score", f"{score:.3f}")
        st.caption(
            f"Decision threshold: {threshold:.4f}. "
            "The score is not a calibrated cancer probability."
        )

with st.expander("Held-out test performance"):
    metrics = config["test_metrics"]
    st.write(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    st.write(f"Balanced accuracy: {metrics['balanced_accuracy']:.1%}")
    st.write(f"Malignant recall: {metrics['malignant_recall']:.1%}")
    st.write(f"Benign recall: {metrics['benign_recall']:.1%}")
    st.caption(
        f"Evaluated on {metrics['images']} images from "
        f"{metrics['patient_groups']} held-out patient groups."
    )