# ML Assignment 2 — Credit Card Default Classification

**Course:** M.Tech (AIML/DSE) — Machine Learning
**Submission for:** Assignment 2 (15 Marks)

---

## a. Problem Statement

Predicting which credit card clients are likely to default on their payment is a core risk-management problem for any bank or card issuer — catching high-risk clients early lets the bank adjust credit limits or follow up before a missed payment happens. The goal of this assignment is to build and compare multiple classification models that predict, from a client's demographic profile, credit limit, and their last 6 months of billing and repayment history, **whether that client will default on their payment next month (Yes) or not (No)** — a binary classification problem.

## b. Dataset Description

- **Name:** Default of Credit Card Clients
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh, I. C., & Lien, C. H., 2009 — originally collected from a bank in Taiwan; also available on Kaggle as "Default of Credit Card Clients Dataset")
- **Instances:** 30,000 credit card clients (assignment minimum: 500 ✅ — 60x over)
- **Features:** 23 predictive features after dropping the unique `ID` column (assignment minimum: 12 ✅):
  - **Demographic:** SEX, EDUCATION, MARRIAGE, AGE
  - **Credit information:** LIMIT_BAL (credit limit)
  - **Repayment history (6 months):** PAY_1 – PAY_6 (repayment status each month)
  - **Billing history (6 months):** BILL_AMT1 – BILL_AMT6 (bill statement amount each month)
  - **Payment history (6 months):** PAY_AMT1 – PAY_AMT6 (amount paid each month)
- **Target:** `Default` (Yes / No) — binary classification. Class balance is ~22% Yes (will default) / 78% No, moderately imbalanced, which is why MCC and F1 matter here in addition to Accuracy.
- **Cleaning applied:**
  - Dropped `ID` (unique identifier, not predictive).
  - Renamed `PAY_0` → `PAY_1` so the six repayment-status columns are consistently numbered.
  - `EDUCATION` and `MARRIAGE` contained a few category codes not listed in the official UCI codebook (`EDUCATION`: 0, 5, 6; `MARRIAGE`: 0). Instead of deleting those ~400 rows, they were folded into the existing "others" category (`EDUCATION=4`, `MARRIAGE=3`) so the full 30,000-row sample is preserved.

## c. GitHub Repository Link

**`[PASTE YOUR GITHUB REPO LINK HERE AFTER YOU PUSH THE CODE]`**
*(See the setup instructions provided separately for how to create this repo and get the link.)*

## d. Models Used

All 5 models below were trained on the **same dataset**, using the **same 80/20 train-test split** (`random_state=42`, stratified) and the **same preprocessing** (StandardScaler on the 20 numeric/ordinal features, OneHotEncoder on the 3 categorical features — SEX, EDUCATION, MARRIAGE), so the comparison is fair.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8088 | 0.7100 | 0.6923 | 0.2442 | 0.3610 | 0.3302 |
| Decision Tree | 0.8157 | 0.7441 | 0.6471 | 0.3662 | 0.4678 | 0.3882 |
| kNN | 0.8088 | 0.7304 | 0.6236 | 0.3421 | 0.4418 | 0.3603 |
| Naive Bayes | 0.6607 | 0.7248 | 0.3590 | 0.6805 | 0.4701 | 0.2822 |
| Random Forest (Ensemble) | 0.8167 | 0.7752 | 0.6601 | 0.3527 | 0.4597 | 0.3871 |

*(These exact numbers are reproduced by running `train_models.ipynb` or `train_models.py` — see the reproducibility instructions provided separately.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest **Precision (0.6923)** — when it predicts a default, it's usually right — but the lowest Recall (0.2442) of the strong models, meaning it misses a lot of actual defaulters. The linear decision boundary is conservative on this imbalanced target. |
| Decision Tree | Second-best Accuracy (0.8157) and AUC (0.7441), with noticeably better Recall than Logistic Regression. A single tree is more flexible than a linear model here, but still less stable than an ensemble. |
| kNN | Sits in the middle across every metric. Benefits from the `StandardScaler` step (distance-based models need scaled inputs) but is affected by the "curse of dimensionality" from one-hot encoding plus 20 numeric columns. |
| Naive Bayes | Lowest Accuracy and Precision, but by far the **highest Recall (0.6805)** — it flags far more clients as high-risk, including many false positives. Its independence assumption is heavily violated here (the 6 `BILL_AMT` columns are highly correlated with each other, as are the 6 `PAY_*` columns), which biases it toward the minority ("will default") class. Useful for a bank that prefers to over-flag risk rather than miss a defaulter. |
| Random Forest (Ensemble) | **Best overall** — highest Accuracy (0.8167) and AUC (0.7752). Averaging 300 decision trees smooths out the overfitting seen in the single Decision Tree, giving the most reliable all-round performance. |
| **Overall Winner for this dataset** | **Random Forest (Ensemble)** — best Accuracy and AUC, and a strong MCC, making it the most balanced choice overall. For a bank that specifically wants to **catch more future defaulters even at the cost of false alarms**, Naive Bayes' much higher Recall would be the pragmatic choice instead. |

---

## Repository Structure

```
project-folder/
│-- app.py                  # Streamlit app (dataset upload, model selection, metrics, confusion matrix)
│-- requirements.txt        # All dependencies needed to run app.py and the notebook
│-- README.md                # This file
│-- credit_raw.csv          # Full raw dataset (source data, used only to regenerate models)
│-- test_data.csv           # 20% held-out test split (upload this to the Streamlit app)
│-- train_models.py         # Training script (Python .py version)
│-- train_models.ipynb      # Training notebook (Jupyter .ipynb version, step-by-step with explanations)
│-- model/
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- metrics.json         # Machine-readable version of the comparison table above
│   │-- metrics_table.csv    # Same table as CSV
```

## How to Reproduce / Run

Full step-by-step instructions (local Jupyter Notebook setup, running on the BITS Virtual Lab, GitHub push, and Streamlit Community Cloud deployment) are provided separately in the assignment write-up. In short:

```bash
pip install -r requirements.txt
python train_models.py        # regenerates model/*.pkl and test_data.csv
streamlit run app.py          # launches the interactive demo locally
```

## Live Streamlit App Link

**`[PASTE YOUR DEPLOYED STREAMLIT APP LINK HERE]`**

## Streamlit App Features

- **Dataset upload (CSV):** upload `test_data.csv` (or any CSV with the same 23 feature columns, with or without the `Default` column) from the sidebar.
- **Model selection dropdown:** choose any of the 5 trained models, or tick "Compare all 5 models" to see every model's metrics side by side on the same uploaded data.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC are computed live on whatever data is uploaded (if it has a `Default` column).
- **Confusion matrix & classification report:** shown for the selected model, computed live on the uploaded data.
