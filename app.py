"""
app.py - Streamlit App for ML Assignment 2
--------------------------------------------
Problem : Predict credit card client default (binary classification: Yes/No)

Required features implemented here:
  a. Dataset upload option (CSV)              -> st.file_uploader
  b. Model selection dropdown                 -> st.selectbox
  c. Display of evaluation metrics            -> metrics table
  d. Confusion matrix / classification report -> heatmap + sklearn report

The app loads the 5 pre-trained pipelines (preprocessing + model bundled
together) saved by train_models.py, applies the chosen model to whatever
CSV the user uploads, and displays the results. Because each .pkl file is
a full sklearn Pipeline (ColumnTransformer + classifier), the app never has
to re-implement preprocessing logic -> this avoids train/serve skew.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Card Default - Model Demo", layout="wide")
st.title("💳 Credit Card Default - Classification Model Demo")
st.caption(
    "M.Tech (AIML/DSE) - Machine Learning - Assignment 2. "
    "Upload the test CSV, pick a model, and see how it performs."
)

# ---------------------------------------------------------------------------
# Model registry: maps a friendly name -> saved pipeline file
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
}


@st.cache_resource
def load_model(path: str):
    """Load a trained pipeline from disk. Cached so repeated selections
    don't reload the file from scratch on every rerun."""
    return joblib.load(path)


def compute_metrics(y_true, y_pred, y_proba):
    """Return the 6 metrics required by the assignment as a dict."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Sidebar controls (feature a + b)
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Upload test_data.csv from the repo, or any CSV with the same "
         "columns. Include the 'Default' column if you want metrics computed."
)
model_choice = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
compare_all = st.sidebar.checkbox("Compare all 5 models on this data", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models implemented:** Logistic Regression, Decision Tree, kNN, "
    "Naive Bayes (Gaussian), Random Forest (Ensemble)"
)

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started "
            "(use the provided test_data.csv to reproduce the reported results).")
    st.stop()

data = pd.read_csv(uploaded_file)
st.subheader("Preview of uploaded data")
st.dataframe(data.head(), use_container_width=True)

has_labels = "Default" in data.columns
if has_labels:
    X_new = data.drop(columns=["Default"])
    y_new = data["Default"].map({"Yes": 1, "No": 0})
else:
    X_new = data.copy()
    st.warning(
        "No 'Default' column found - showing predictions only. "
        "Upload data with a 'Default' column to see evaluation metrics."
    )


def run_single_model(name, path, X_new, y_new, has_labels):
    st.subheader(f"Results - {name}")
    pipe = load_model(path)
    preds = pipe.predict(X_new)
    proba = pipe.predict_proba(X_new)[:, 1]

    pred_labels = pd.Series(preds).map({1: "Yes", 0: "No"})
    result_df = data.copy()
    result_df["Predicted_Default"] = pred_labels.values
    result_df["Default_Probability"] = proba.round(3)
    st.dataframe(result_df.head(20), use_container_width=True)

    if not has_labels:
        return

    # ---- Evaluation metrics (feature c) ----
    metrics = compute_metrics(y_new, preds, proba)
    st.markdown("**Evaluation Metrics**")
    metric_cols = st.columns(6)
    for col, (k, v) in zip(metric_cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    # ---- Confusion matrix + classification report (feature d) ----
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_new, preds)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_new, preds, target_names=["No", "Yes"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    return metrics


if compare_all:
    st.subheader("Model Comparison on Uploaded Data")
    if not has_labels:
        st.warning("Upload data with a 'Default' column to compare models.")
    else:
        rows = []
        for name, path in MODEL_FILES.items():
            pipe = load_model(path)
            preds = pipe.predict(X_new)
            proba = pipe.predict_proba(X_new)[:, 1]
            m = compute_metrics(y_new, preds, proba)
            m["Model"] = name
            rows.append(m)
        comp_df = pd.DataFrame(rows).set_index("Model")[
            ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
        ]
        st.dataframe(comp_df.round(4), use_container_width=True)
        st.bar_chart(comp_df)
else:
    run_single_model(model_choice, MODEL_FILES[model_choice], X_new, y_new if has_labels else None, has_labels)

st.markdown("---")
st.caption("Assignment 2 - Machine Learning - BITS Pilani WILP (M.Tech AIML/DSE)")
