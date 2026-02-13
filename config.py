DATA_PATH = "data/loan_approval_dataset.csv"
TARGET_COL = "loan_status"
SAVE_DIR = "models/saved"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# COMMON CLEANUP
# -----------------------------
DROP_COLS_IF_PRESENT = ["Unnamed: 0", "Loan_ID", "loan_id", "id"]

# -----------------------------
# MODEL FILES (for Streamlit)
# -----------------------------
MODEL_FILES = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "dt.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "nb.pkl",
    "Random Forest (Ensemble)": "rf.pkl",
    "XGBoost (Ensemble)": "xgb.pkl"
}

# -----------------------------
# SAVED ARTIFACTS
# -----------------------------
LABEL_ENCODER_FILE = "label_encoder.pkl"
METRICS_FILE = "metrics.csv"