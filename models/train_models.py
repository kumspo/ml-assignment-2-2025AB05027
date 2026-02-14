import os
import pandas as pd
import numpy as np
import joblib
import traceback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from datetime import datetime

from config import DATA_PATH, TARGET_COL, SAVE_DIR, RANDOM_STATE, TEST_SIZE
from models.utils import evaluate_model, clean_dataframe, split_X_y,to_dense

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)


    # Drop unwanted ID/index columns if present
    df = clean_dataframe(df)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    X, y = split_X_y(df)

    print(f"Minimum Feature Size: {X.shape[1]}")
    print(f"Minimum Instance Size: {X.shape[0]}")

    # Save training feature column order (important for Streamlit inference)
    feature_cols_path = os.path.join(SAVE_DIR, "feature_columns.pkl")
    joblib.dump(list(X.columns), feature_cols_path)
    print(f"✅ Saved feature columns -> {feature_cols_path}")

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)
    print("Target classes:", list(label_encoder.classes_))

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    # Models
    models = {
        "logistic": LogisticRegression(max_iter=5000,solver="liblinear",
                                       penalty="l2",
                                       C=1.0,
                                       class_weight="balanced"),
        "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=15,weights="distance", metric="manhattan"),
        "nb": GaussianNB(),
        "rf": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "xgb": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )
    }

    os.makedirs(SAVE_DIR, exist_ok=True)

    metrics_table = []
    failed_models = []


    for name, model in models.items():
        print(f"\n==============================")
        print(f"Training: {name}")
        print(f"==============================")

        try:
            # NB needs dense array (because onehot produces sparse matrix)
            if name == "nb":
                clf = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("to_dense", FunctionTransformer(to_dense)),
                    ("model", model)
                ])
            else:
                clf = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # For AUC
            y_prob = None
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    y_prob = probs[:, 1]
                else:
                    y_prob = probs

            results = evaluate_model(y_test, y_pred, y_prob)
            metrics_table.append({"Model": name, **results})

            # Save pipeline model
            save_path = os.path.join(SAVE_DIR, f"{name}.pkl")
            joblib.dump(clf, save_path)
            print(f"✅ Saved -> {save_path}")
        except Exception as e:
            print(f"❌ Model failed: {name}")
            print(f"   Error: {e}")

            failed_models.append({"Model": name, "Error": str(e)})
            # Save full traceback log
            log_path = os.path.join(SAVE_DIR, "training_errors.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n==============================\n")
                f.write(f"MODEL FAILED: {name}\n")
                f.write(f"TIME: {datetime.now()}\n")
                f.write(str(e) + "\n\n")
                f.write(traceback.format_exc())
                f.write("\n")
            continue

    if failed_models:
        print("\n==============================")
        print("⚠️ MODELS THAT FAILED")
        print("==============================")
        for item in failed_models:
            print(f"- {item['Model']} : {item['Error']}")
    else:
        print("\n✅ All models trained successfully!")

    # Save label encoder
    joblib.dump(label_encoder, os.path.join(SAVE_DIR, "label_encoder.pkl"))

    # Save metrics table
    if len(metrics_table) > 0:
        metrics_df = pd.DataFrame(metrics_table)
        metrics_df = metrics_df[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
        metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)
        print("\n==============================")
        print("FINAL METRICS TABLE")
        print("==============================")
        print(metrics_df)
    else:
        print("❌ No models succeeded. metrics.csv not created.")

    print("\nDone. All models saved inside models/saved/")

if __name__ == "__main__":
    main()