"""
train_models.py
----------------
ML Assignment 2 - Model Training Script
Dataset : Default of Credit Card Clients (UCI Machine Learning Repository)
Task    : Binary Classification - Predict whether a credit card client will
          default on their payment next month.

This script:
1. Loads and cleans the raw dataset
2. Splits it into train (80%) and test (20%) sets
3. Saves the test split as test_data.csv (this is what gets uploaded to the Streamlit app)
4. Builds a preprocessing + model Pipeline for each of the 5 required classifiers
5. Trains every pipeline on the training set
6. Evaluates every pipeline on the test set using the 6 required metrics
7. Saves each trained pipeline (preprocessing + model bundled together) as a .pkl file
   inside model/ so the Streamlit app can load and use it directly on raw CSV data
8. Writes a metrics comparison table to model/metrics.json and model/metrics_table.csv
   (used to fill the README.md comparison table)
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Step 1: Load and clean the data
# ---------------------------------------------------------------------------
print("Step 1: Loading and cleaning data ...")
df = pd.read_csv("credit_raw.csv")

# ID is a unique identifier, not a predictive feature -> drop it
df = df.drop(columns=["ID"])

# Rename PAY_0 -> PAY_1 so the 6 repayment-status columns are consistently
# named PAY_1..PAY_6 (the raw file skips straight from PAY_0 to PAY_2 - a
# known quirk of this dataset).
df = df.rename(columns={"PAY_0": "PAY_1"})

# EDUCATION and MARRIAGE contain a few undocumented category codes not
# listed in the official UCI codebook (EDUCATION: 0, 5, 6 ; MARRIAGE: 0).
# Rather than dropping ~400 rows, we fold these undocumented codes into the
# existing "others" bucket (EDUCATION=4, MARRIAGE=3) so we keep the full
# 30,000-row sample while still cleaning up the categories.
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# Target column: rename for clarity and keep it numeric (1 = will default,
# 0 = will not default)
df = df.rename(columns={"default.payment.next.month": "Default"})

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Features: {df.shape[1] - 1}  |  Instances: {df.shape[0]}")
print(f"Class balance:\n{df['Default'].value_counts(normalize=True)}\n")

# ---------------------------------------------------------------------------
# Step 2: Train / Test split (test set doubles as the CSV used by the app)
# ---------------------------------------------------------------------------
print("Step 2: Splitting into train (80%) and test (20%) sets ...")
X = df.drop(columns=["Default"])
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Save the RAW test data (features + true label, human readable Yes/No) as
# test_data.csv -- this is the file required by the assignment ("test data
# used in your experiments") and is also what gets uploaded to the
# Streamlit app for live evaluation, so results shown in the app match
# what's reported here.
test_export = X_test.copy()
test_export["Default"] = y_test.map({1: "Yes", 0: "No"})
test_export.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_export)} rows\n")

# ---------------------------------------------------------------------------
# Step 3: Build the shared preprocessing pipeline
# ---------------------------------------------------------------------------
# SEX, EDUCATION, MARRIAGE are unordered categorical codes -> one-hot encode.
# Everything else (credit limit, age, 6 months of repayment status, bill
# amounts, payment amounts) is numeric/ordinal -> scale it.
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}\n")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# ---------------------------------------------------------------------------
# Step 4: Define the 5 required models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE
    ),
}

filename_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

# ---------------------------------------------------------------------------
# Step 5: Train, evaluate and save each model
# ---------------------------------------------------------------------------
os.makedirs("model", exist_ok=True)
results = []
print("Step 3-5: Training and evaluating each model ...\n")

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    # Save the FULL pipeline (preprocessing + trained model) so the Streamlit
    # app can call .predict() / .predict_proba() directly on raw CSV data
    out_path = f"model/{filename_map[name]}"
    joblib.dump(pipe, out_path)

    print(f"{name:28s} | Acc={metrics['Accuracy']:.4f}  AUC={metrics['AUC']:.4f}  "
          f"Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}  "
          f"F1={metrics['F1']:.4f}  MCC={metrics['MCC']:.4f}  -> saved to {out_path}")

# ---------------------------------------------------------------------------
# Step 6: Persist comparison table for README / app
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model/metrics_table.csv", index=False)
with open("model/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll models trained and saved to model/")
print("\nComparison Table:\n")
print(results_df.to_string(index=False))
