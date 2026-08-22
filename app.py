import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# ---------- Load model and feature stats ----------
model = joblib.load("fetal_health_model.pkl")

with open("feature_stats.json", "r") as f:
    stats = json.load(f)

median = stats["median"]
fmin = stats["min"]
fmax = stats["max"]

# Full feature list, in the exact order the model was trained on
FEATURES = list(median.keys())

# Top features the user can adjust — based on feature importance / MI scores
TOP_FEATURES = [
    "mean_value_of_short_term_variability",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "histogram_mean",
    "percentage_of_time_with_abnormal_long_term_variability",
    "accelerations",
]

CLASS_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}
CLASS_COLORS = {1: "🟢", 2: "🟡", 3: "🔴"}

# ---------- Page setup ----------
st.set_page_config(page_title="Fetal Health Risk Classifier", page_icon="🩺")

st.title("🩺 Fetal Health Risk Classifier")
st.write(
    "Predicts fetal health status (Normal / Suspect / Pathological) from "
    "cardiotocography (CTG) measurements, using an XGBoost model trained on "
    "the UCI Fetal Health dataset."
)

st.markdown(
    "**This tool is for demonstration purposes only. It is not a medical "
    "device and should never be used to guide real clinical decisions.**"
)

st.divider()
st.subheader("Adjust key CTG measurements")
st.caption(
    "These are the features most influential to the model's predictions. "
    "All other features are set to their typical (median) dataset values."
)

# ---------- Build input form for top features ----------
user_inputs = {}

for feature in TOP_FEATURES:
    lo = float(fmin[feature])
    hi = float(fmax[feature])
    default = float(median[feature])

    # Use finer step size for features with small value ranges
    step = (hi - lo) / 100 if (hi - lo) > 0 else 0.01

    label = feature.replace("_", " ").title()
    user_inputs[feature] = st.slider(
        label, min_value=lo, max_value=hi, value=default, step=step
    )

st.divider()

# ---------- Predict ----------
if st.button("Predict Fetal Health Status", type="primary"):
    # Build full feature vector: user inputs for top features, median for the rest
    row = []
    for feature in FEATURES:
        if feature in user_inputs:
            row.append(user_inputs[feature])
        else:
            row.append(median[feature])

    X_input = pd.DataFrame([row], columns=FEATURES)

    # Model was trained on labels shifted to 0,1,2 — shift prediction back to 1,2,3
    pred_shifted = model.predict(X_input)[0]
    pred_class = int(pred_shifted) + 1

    proba = model.predict_proba(X_input)[0]

    st.subheader("Prediction")
    st.markdown(f"### {CLASS_COLORS[pred_class]} {CLASS_LABELS[pred_class]}")

    st.write("Class probabilities:")
    proba_df = pd.DataFrame({
        "Class": [CLASS_LABELS[1], CLASS_LABELS[2], CLASS_LABELS[3]],
        "Probability": proba
    })
    st.bar_chart(proba_df.set_index("Class"))

    if pred_class == 3:
        st.warning(
            "This result suggests a Pathological classification. In a real "
            "clinical context, this would warrant immediate escalation to a "
            "healthcare professional — never treat this tool's output as a diagnosis."
        )

st.divider()
st.caption(
    "Built as part of the Intelligent Fetal Health Classification project. "
    "See the full analysis notebook and discussion of limitations in the GitHub repo."
)
