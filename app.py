import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix

import config
from config import (
    SAVE_DIR, TARGET_COL,
    MODEL_FILES, LABEL_ENCODER_FILE, METRICS_FILE
)
from models.utils import evaluate_model, clean_dataframe


st.set_page_config(page_title="ML Assignment 2 - Classification Models", layout="wide")
st.title("📌 ML Assignment 2 - Classification Models Demo - Loan Approval Prediction")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_resource
def load_label_encoder():
    path = os.path.join(SAVE_DIR, LABEL_ENCODER_FILE)
    return joblib.load(path)


@st.cache_data
def load_metrics_table():
    path = os.path.join(SAVE_DIR, METRICS_FILE)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Controls")

model_name = st.sidebar.selectbox("Select a Model", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload CSV (Test Data)", type=["csv"])

st.sidebar.info(
    "📌 Upload a CSV file containing the same feature columns as training data.\n\n"
    f"⚠️ Do NOT include the target column ({TARGET_COL}) in uploaded test CSV."
)

# -----------------------------
# SHOW TRAINING METRICS TABLE
# -----------------------------
st.subheader("📊 Model Comparison Table (from Training)")

metrics_df = load_metrics_table()
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.warning("metrics.csv not found. Train models first using: python models/train_models.py")


# -----------------------------
# MAIN PREDICTION SECTION
# -----------------------------
st.subheader("🧪 Test Your Model with Uploaded Data")

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin predictions.")
    st.stop()

# Read uploaded CSV
try:
    test_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading uploaded CSV: {e}")
    st.stop()

# Drop unwanted columns if present
test_df = clean_dataframe(test_df)

# If user mistakenly uploaded target column
if TARGET_COL in test_df.columns:
    st.error(
        f"Your uploaded CSV contains the target column '{TARGET_COL}'.\n"
        "Please upload test data WITHOUT the target column."
    )
    st.stop()

st.write("✅ Uploaded data preview:")
st.dataframe(test_df.head(), use_container_width=True)



# Load model + encoder
try:
    model_path = os.path.join(SAVE_DIR, MODEL_FILES[model_name])
    model = load_model(model_path)
    label_encoder = load_label_encoder()
except Exception as e:
    st.error(f"Error loading saved models. Train first.\n\nDetails: {e}")
    st.stop()

@st.cache_resource
def load_feature_columns():
    path = os.path.join(SAVE_DIR, "feature_columns.pkl")
    return joblib.load(path)

try:
    expected_cols = load_feature_columns()

    missing = [c for c in expected_cols if c not in test_df.columns]
    extra = [c for c in test_df.columns if c not in expected_cols]
    if missing:
        st.error(f"Missing columns in uploaded CSV: {missing}")
        st.stop()

    if extra:
        st.warning(f"Extra columns ignored: {extra}")

    #enforce same column order as training
    test_df = test_df[expected_cols]

except Exception as e:
    st.warning(f"⚠️ Could not load feature_columns.pkl. Skipping strict validation.\n\nDetails: {e}")

# Predict
try:
    y_pred_encoded = model.predict(test_df)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
except Exception as e:
    st.error(
        "Prediction failed.\n\n"
        "Most likely your uploaded CSV columns don't match training columns.\n\n"
        f"Details: {e}"
    )
    st.stop()

# Display predictions
pred_df = pd.DataFrame({"Prediction": y_pred_labels})
st.subheader("✅ Predictions")
st.dataframe(pred_df, use_container_width=True)

# Download predictions
csv_out = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Predictions as CSV",
    data=csv_out,
    file_name="predictions.csv",
    mime="text/csv"
)

# -----------------------------
# OPTIONAL: Evaluate if labels provided
# -----------------------------
st.subheader("📌 Evaluate on Test Labels (Optional)")
st.write("Upload a separate labels CSV to evaluate metrics + confusion matrix.")

true_labels_file = st.file_uploader("Upload True Labels CSV (Optional)", type=["csv"])

if true_labels_file is not None:
    try:
        y_true_df = pd.read_csv(true_labels_file)
    except Exception as e:
        st.error(f"Error reading labels file: {e}")
        st.stop()

    # Detect label column
    if TARGET_COL in y_true_df.columns:
        y_true_raw = y_true_df[TARGET_COL]
    else:
        if y_true_df.shape[1] == 1:
            y_true_raw = y_true_df.iloc[:, 0]
        else:
            st.error(
                f"Could not find '{TARGET_COL}' in label file.\n"
                "Either include the target column or upload a CSV with only 1 label column."
            )
            st.stop()

    # Encode y_true
    try:
        y_true_encoded = label_encoder.transform(y_true_raw)
    except Exception:
        st.error(
            "True labels do not match training label classes.\n"
            f"Expected classes: {list(label_encoder.classes_)}"
        )
        st.stop()

    # Probabilities for AUC
    y_prob = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(test_df)
        if len(np.unique(y_true_encoded)) == 2:
            y_prob = probs[:, 1]
        else:
            y_prob = probs

    results = evaluate_model(y_true_encoded, y_pred_encoded, y_prob)

    st.subheader("📌 Evaluation Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{results['Accuracy']:.4f}")
    col1.metric("AUC", f"{results['AUC']:.4f}" if not np.isnan(results["AUC"]) else "N/A")

    col2.metric("Precision", f"{results['Precision']:.4f}")
    col2.metric("Recall", f"{results['Recall']:.4f}")

    col3.metric("F1 Score", f"{results['F1']:.4f}")
    col3.metric("MCC", f"{results['MCC']:.4f}")

    st.subheader("🧾 Confusion Matrix")
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("📄 Classification Report")
    report = classification_report(
        y_true_encoded,
        y_pred_encoded,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    st.text(report)