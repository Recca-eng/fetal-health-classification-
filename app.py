import streamlit as st
import joblib
import json
import pandas as pd

# ---------- Load model, feature stats, and real example cases ----------
model = joblib.load("fetal_health_model.pkl")

with open("feature_stats.json", "r") as f:
    stats = json.load(f)

with open("sample_cases.json", "r") as f:
    sample_cases = json.load(f)

median = stats["median"]
fmin = stats["min"]
fmax = stats["max"]

FEATURES = list(median.keys())

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

# ---------- Session state: holds the currently active full feature vector ----------
if "current_values" not in st.session_state:
    st.session_state.current_values = dict(median)  # start at dataset medians

# ---------- Example case buttons ----------
st.subheader("Try a real example")
st.caption(
    "These are real cases from the model's test set, one per class, so you can "
    "see all three outcomes without needing to guess slider values."
)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🟢 Load Normal Example"):
        st.session_state.current_values = dict(sample_cases["normal"])
with col2:
    if st.button("🟡 Load Suspect Example"):
        st.session_state.current_values = dict(sample_cases["suspect"])
with col3:
    if st.button("🔴 Load Pathological Example"):
        st.session_state.current_values = dict(sample_cases["pathological"])

st.divider()
st.subheader("Or adjust key CTG measurements manually")
st.caption(
    "These are the features most influential to the model's predictions. "
    "All other features stay fixed at their currently loaded values."
)

# ---------- Build sliders for top features, seeded from session state ----------
user_inputs = {}

for feature in TOP_FEATURES:
    lo = float(fmin[feature])
    hi = float(fmax[feature])
    current = float(st.session_state.current_values.get(feature, median[feature]))
    # clamp in case a loaded example sits at the exact dataset extremes
    current = min(max(current, lo), hi)
    step = (hi - lo) / 100 if (hi - lo) > 0 else 0.01

    label = feature.replace("_", " ").title()
    user_inputs[feature] = st.slider(
        label, min_value=lo, max_value=hi, value=current, step=step, key=f"slider_{feature}"
    )

# update session state with any manual slider moves
st.session_state.current_values.update(user_inputs)

st.divider()

# ---------- Predict ----------
if st.button("Predict Fetal Health Status", type="primary"):
    row = [st.session_state.current_values.get(f, median[f]) for f in FEATURES]
    X_input = pd.DataFrame([row], columns=FEATURES)

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
