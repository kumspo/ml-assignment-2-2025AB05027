import numpy as np
import pandas as pd
from config import DROP_COLS_IF_PRESENT, TARGET_COL

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Returns all required metrics in a dictionary.
    Uses weighted avg for multiclass metrics.
    """
    results = {}

    results["Accuracy"] = accuracy_score(y_true, y_pred)
    results["Precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    results["Recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    results["F1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    results["MCC"] = matthews_corrcoef(y_true, y_pred)

    # AUC
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                results["AUC"] = roc_auc_score(y_true, y_prob)
            else:
                results["AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except Exception:
            results["AUC"] = np.nan
    else:
        results["AUC"] = np.nan

    return results


def get_confusion_matrix(y_true, y_pred):
    """
    Returns confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)

# -----------------------------
# CLEAN DATAFRAME (Reusable)
# -----------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unwanted ID-like columns if present.
    Returns a cleaned copy of the dataframe.
    """
    df = df.copy()

    for col in DROP_COLS_IF_PRESENT:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

# -----------------------------
# SPLIT FEATURES + TARGET
# -----------------------------
def split_X_y(df: pd.DataFrame):
    """
    Splits dataframe into X (features) and y (target) using TARGET_COL.

    Returns:
        X (pd.DataFrame), y (pd.Series)

    Raises:
        ValueError if TARGET_COL is missing.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset!")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y