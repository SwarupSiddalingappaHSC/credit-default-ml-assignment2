# ML Assignment 2 — Credit Card Default Classification

**Course:** M.Tech (AIML/DSE) — Machine Learning
**Submission for:** Assignment 2 (15 Marks)

---

## a. Problem Statement

Predicting which credit card clients are likely to default on their payment is a core risk-management problem for banks and card issuers. Catching high-risk clients early lets a bank adjust credit limits or follow up before a payment is missed. This assignment builds and compares multiple classification models that predict, from a client's demographic profile, credit limit, and their last 6 months of billing and repayment history, whether that client will default on their payment next month. This is a binary classification problem (`Default`: Yes or No).

## b. Dataset Description

- **Name:** Default of Credit Card Clients
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh, I. C., & Lien, C. H., 2009 — originally collected from a bank in Taiwan; also available on Kaggle as "Default of Credit Card Clients Dataset")
- **Instances:** 30,000 credit card clients (assignment minimum is 500, so this is 60x over)
- **Features:** 23 predictive features after dropping the unique `ID` column (assignment minimum is 12):
  - **Demographic:** SEX, EDUCATION, MARRIAGE, AGE
  - **Credit information:** LIMIT_BAL (credit limit)
  - **Repayment history (6 months):** PAY_1 – PAY_6 (repayment status each month)
  - **Billing history (6 months):** BILL_AMT1 – BILL_AMT6 (bill statement amount each month)
  - **Payment history (6 months):** PAY_AMT1 – PAY_AMT6 (amount paid each month)
- **Target:** `Default` (Yes / No), binary classification. Class balance is ~22% Yes (will default) and 78% No. The data is moderately imbalanced, so MCC and F1 are tracked alongside Accuracy.
- **Cleaning applied:**
  - Dropped `ID` (unique identifier, not predictive).
  - Renamed `PAY_0` → `PAY_1` so the six repayment-status columns are consistently numbered.
  - `EDUCATION` and `MARRIAGE` contained a few category codes not listed in the official UCI codebook (`EDUCATION`: 0, 5, 6; `MARRIAGE`: 0). Instead of deleting those ~400 rows, they were folded into the existing "others" category (`EDUCATION=4`, `MARRIAGE=3`) so the full 30,000-row sample is preserved.

## c. GitHub Repository Link

[credit-default-ml-assignment2](https://github.com/SwarupSiddalingappaHSC/credit-default-ml-assignment2)

## d. Models Used

All 5 models were trained on the same dataset, with the same 80/20 train-test split (`random_state=42`, stratified) and the same preprocessing: StandardScaler on the 20 numeric/ordinal features, OneHotEncoder on the 3 categorical features (SEX, EDUCATION, MARRIAGE). This keeps the comparison fair.

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
| Logistic Regression | Precision is the best of all 5 models at 0.6923, so its default predictions are usually correct. Recall is weak at 0.2442, though, so it misses most actual defaulters. The linear boundary stays conservative on this imbalanced target. |
| Decision Tree | Accuracy (0.8157) and AUC (0.7441) are the second-best in the group. Recall is noticeably better than Logistic Regression. A single tree fits the data more flexibly than a linear model, though it's still less stable than an ensemble of trees. |
| kNN | Metrics land in the middle of the pack across the board. The `StandardScaler` step helps, since distance-based models need scaled inputs, but the curse of dimensionality (20 numeric columns plus one-hot encoded categories) still limits performance. |
| Naive Bayes | Accuracy and Precision are the lowest of the 5 models, but Recall is the highest at 0.6805. It flags many more clients as high-risk, including a lot of false positives. This happens because the independence assumption breaks down badly here: the 6 `BILL_AMT` columns are correlated with each other, and so are the 6 `PAY_*` columns. For a bank that would rather over-flag risk than miss a defaulter, this model has an advantage. |
| Random Forest (Ensemble) | Accuracy (0.8167) and AUC (0.7752) are the best of the 5 models. Averaging 300 decision trees smooths out the overfitting a single tree shows, and gives the most reliable performance overall. |
| **Overall Winner for this dataset** | **Random Forest (Ensemble)**, based on the best Accuracy, AUC, and a strong MCC. If the priority shifts to catching as many future defaulters as possible, even at the cost of more false alarms, Naive Bayes' much higher Recall makes it the more practical pick instead. |

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

[Open the deployed Streamlit app](https://credit-default-ml-assignment2-g9s9j3ghdsrebqqb6zwjmu.streamlit.app/)

## Streamlit App Features

- **Dataset upload (CSV):** upload `test_data.csv` (or any CSV with the same 23 feature columns, with or without the `Default` column) from the sidebar.
- **Model selection dropdown:** choose any of the 5 trained models, or tick "Compare all 5 models" to see every model's metrics side by side on the same uploaded data.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC are computed live on whatever data is uploaded (if it has a `Default` column).
- **Confusion matrix & classification report:** shown for the selected model, computed live on the uploaded data.
