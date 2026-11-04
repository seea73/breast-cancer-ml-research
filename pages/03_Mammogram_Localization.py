
from pathlib import Path
import json

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Mammogram Localization",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT_DIR / "module3_mammogram"
MODELS_DIR = MODULE_DIR / "models"

MODEL_FILES = {
    "U-Net v1 (Baseline)": "mammogram_unet_best.keras",
    "U-Net v3 (Experimental)": "mammogram_unet_v3_best.keras",
}

# Results measured on the validation set, not the test set.
VALIDATION_METRICS = {
    "U-Net v1 (Baseline)": {
        "dice": 0.251053,
        "iou": 0.180607,
        "missed": 8,
        "false_highlights": 21,
        "false_positive_pixels": 9878,
    },
    "U-Net v3 (Experimental)": {
        "dice": 0.258160,
        "iou": 0.184672,
        "missed": 6,
        "false_highlights": 20,
        "false_positive_pixels": 8478,
    },
}

THRESHOLD = 0.15
IMAGE_SIZE = 256


@st.cache_resource
def load_model(model_filename):
    model_path = MODELS_DIR / model_filename
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


st.title("Mammogram Region Localization")

st.write(
    "Explore mammogram region localization using "
    "experimental U-Net models trained on mini-MIAS."
)

st.warning(
    "Educational research demonstration only. These models "
    "have limited localization accuracy, frequently highlight "
    "normal tissue, and can completely miss annotated abnormalities. "
    "They must not be used for diagnosis, screening, or clinical decisions."
)

selected_model = st.selectbox(
    "Select localization model",
    options=list(MODEL_FILES.keys()),
    index=1
)

metrics = VALIDATION_METRICS[selected_model]

try:
    model = load_model(MODEL_FILES[selected_model])
except Exception as error:
    st.error(f"Could not load the selected model: {error}")
    st.stop()

st.subheader("Validation performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Mean abnormal Dice",
    f"{metrics['dice']:.3f}"
)

col2.metric(
    "Mean abnormal IoU",
    f"{metrics['iou']:.3f}"
)

col3.metric(
    "Normal images with highlights",
    f"{metrics['false_highlights']}/32"
)

st.caption(
    f"Validation set: 48 images (16 abnormal, 32 normal). "
    f"Completely missed abnormalities: {metrics['missed']}/16. "
    f"Total false-positive pixels on normal images: "
    f"{metrics['false_positive_pixels']:,}. "
    f"Decision threshold: {THRESHOLD:.2f}. "
    "Both models are evaluated at the same threshold. "
    "These are validation results, not independent test results."
)

uploaded = st.file_uploader(
    "Upload a mammogram",
    type=["pgm", "png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is not None:
    try:
        with Image.open(uploaded) as source:
            if source.mode not in {"L", "RGB", "RGBA"}:
                st.error(
                    "Use an 8-bit grayscale or RGB image. "
                    "Higher-bit-depth images require a "
                    "separate preprocessing workflow."
                )
                st.stop()

            image = source.convert("L")
            original_size = image.size

            resized = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR
            )

        if original_size != (1024, 1024):
            st.info(
                "This image differs from the 1024 x 1024 "
                "mini-MIAS format used for training. "
                "Results may be less reliable."
            )

        image_array = (
            np.asarray(resized, dtype=np.float32) / 255.0
        )

        model_input = image_array[None, ..., None]

        with st.spinner("Generating localization mask..."):
            scores = model(
                model_input,
                training=False
            ).numpy()[0, ..., 0]

        mask = scores >= THRESHOLD

        grayscale_rgb = np.repeat(
            image_array[..., None],
            3,
            axis=-1
        )

        overlay = grayscale_rgb.copy()

        overlay[mask] = (
            0.55 * overlay[mask]
            + 0.45 * np.array([1.0, 0.85, 0.0])
        )

        left, right = st.columns(2)

        with left:
            st.image(
                resized,
                caption="Mammogram at model input resolution",
                use_container_width=True
            )

        with right:
            st.image(
                np.clip(overlay, 0, 1),
                caption=(
                    f"{selected_model}: yellow-highlighted regions"
                ),
                use_container_width=True
            )

        st.write(
            f"Highlighted image area: **{mask.mean():.2%}**"
        )

        st.caption(
            "Pixel scores are not calibrated cancer probabilities. "
            "Highlighted regions do not establish malignancy, "
            "and absence of highlights does not exclude disease. "
            "Uploaded images are not evaluated against annotations."
        )

    except Exception as error:
        st.error(f"Could not process the image: {error}")
