from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Analysis",
    page_icon="🔬",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "results" / "cell_classifier.joblib"

st.title("Explainable Breast Cancer Analysis")
st.caption(
    "Educational research demonstration using cell measurements. "
    "Not for clinical diagnosis."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Expected: "
        "results/cell_classifier.joblib"
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)

st.success("Cell-measurement model loaded successfully.")

st.write("**Model:** Logistic regression")
st.write(f"**Saved decision threshold:** {bundle['threshold']:.2f}")
st.write(f"**Required measurements:** {len(bundle['feature_names'])}")

st.info(
    "This module analyzes numerical measurements from breast-mass "
    "samples. It does not accept tissue images or mammograms."
)


import pandas as pd

DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"

if not DATA_PATH.exists():
    st.error("Dataset not found: data/wdbc.data")
    st.stop()

dataset = pd.read_csv(
    DATA_PATH,
    header=None,
    names=["id", "diagnosis"] + bundle["feature_names"]
)

st.divider()
st.subheader("Explore a Dataset Sample")

st.caption(
    "Samples come from the original dataset and include training samples. "
    "This panel demonstrates predictions; it is not a new evaluation."
)

selected_index = st.selectbox(
    "Select a sample",
    options=dataset.index.tolist(),
    format_func=lambda index: (
        f"Row {index} — Sample ID {dataset.loc[index, 'id']}"
    )
)

selected_sample = dataset.loc[
    [selected_index], bundle["feature_names"]
]

probability = bundle["model"].predict_proba(selected_sample)[0, 1]
threshold = bundle["threshold"]

predicted_label = (
    "Malignant" if probability >= threshold else "Benign"
)
actual_label = (
    "Malignant"
    if dataset.loc[selected_index, "diagnosis"] == "M"
    else "Benign"
)

col1, col2, col3 = st.columns(3)

col1.metric("Predicted class", predicted_label)
col2.metric("Model malignant probability", f"{probability:.1%}")
col3.metric("Dataset diagnosis", actual_label)

st.write(f"**Decision threshold:** {threshold:.2f}")

with st.expander("View the sample's 30 measurements"):
    st.dataframe(selected_sample.T.rename(
        columns={selected_index: "Value"}
    ))



import numpy as np
import matplotlib.pyplot as plt

st.subheader("What Influenced This Prediction?")

fitted_pipeline = bundle["model"]
scaler = fitted_pipeline.named_steps["scaler"]
classifier = fitted_pipeline.named_steps["model"]

standardized_values = scaler.transform(selected_sample)[0]

sample_contributions = pd.Series(
    standardized_values * classifier.coef_[0],
    index=bundle["feature_names"]
)

largest_features = sample_contributions.abs().nlargest(10).index
display_contributions = (
    sample_contributions.loc[largest_features].sort_values()
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    display_contributions.index,
    display_contributions.values,
    color=[
        "#E76F51" if value > 0 else "#2A9D8F"
        for value in display_contributions
    ]
)

ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Contribution to malignant log-odds")
ax.set_title("Ten Largest Feature Contributions")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Positive bars push toward malignant; negative bars push toward benign, "
    "relative to the training feature means. These contributions explain "
    "the model calculation, not biological causes."
)

intercept = float(classifier.intercept_[0])
total_log_odds = intercept + float(sample_contributions.sum())

with st.expander("View the prediction calculation"):
    st.write(f"Intercept: {intercept:.4f}")
    st.write(
        f"All 30 feature contributions combined: "
        f"{sample_contributions.sum():.4f}"
    )
    st.write(f"Total log-odds: {total_log_odds:.4f}")
    st.write(f"Model malignant probability: {probability:.4f}")

    from sklearn.metrics import confusion_matrix

st.divider()
st.subheader("Explore the Decision Threshold")

st.caption(
    "This experiment uses out-of-fold training predictions. "
    "It does not change the saved model or its 0.30 threshold."
)

validation_path = PROJECT_ROOT / "results" / "validation_predictions.csv"

if validation_path.exists():
    validation_data = pd.read_csv(validation_path)

    experiment_threshold = st.slider(
        "Validation experiment threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.30,
        step=0.01
    )

    experiment_predictions = (
        validation_data["malignant_probability"]
        >= experiment_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        validation_data["actual"],
        experiment_predictions,
        labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Validation sensitivity", f"{sensitivity:.1%}")
    col2.metric("Validation specificity", f"{specificity:.1%}")
    col3.metric("Missed malignant samples", int(fn))
    col4.metric("False alarms", int(fp))

    st.write(
        "Lower thresholds generally catch more malignant samples "
        "while increasing false alarms."
    )
else:
    st.warning(
        "Save validation_predictions.csv from the notebook "
        "to enable this experiment."
    )

    st.divider()
st.subheader("Held-Out Test Results")

st.caption(
    "These results use 114 held-out samples: 42 malignant and 72 benign. "
    "Both thresholds were fixed before test evaluation. "
    "The validation slider does not change these results."
)

test_results_path = PROJECT_ROOT / "results" / "test_results.csv"
intervals_path = (
    PROJECT_ROOT / "results" / "test_confidence_intervals.csv"
)

if test_results_path.exists():
    saved_test_results = pd.read_csv(
        test_results_path,
        index_col=0
    )

    st.dataframe(
        saved_test_results.style.format({
            "Threshold": "{:.2f}",
            "Accuracy": "{:.1%}",
            "Sensitivity": "{:.1%}",
            "Specificity": "{:.1%}",
            "Precision": "{:.1%}",
            "Missed malignant (FN)": "{:.0f}",
            "False alarms (FP)": "{:.0f}"
        }),
        hide_index=True
    )
else:
    st.warning("Test results file not found.")

if intervals_path.exists():
    saved_intervals = pd.read_csv(
        intervals_path,
        index_col=0
    )

    with st.expander("View 95% confidence intervals"):
        st.dataframe(
            saved_intervals.style.format({
                "Threshold": "{:.2f}",
                "Estimate": "{:.1%}",
                "95% CI lower": "{:.1%}",
                "95% CI upper": "{:.1%}"
            }),
            hide_index=True
        )

st.info(
    "Results come from one split of a small research dataset. "
    "Performance on other populations or clinical settings "
    "has not been established."
)