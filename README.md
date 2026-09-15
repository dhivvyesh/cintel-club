# Loan Approval Prediction — Logistic Regression

A binary classification project that predicts whether a loan application will be **approved** or **rejected**, built on the [Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data) (Kaggle).

The pipeline covers the full ML workflow: load → preprocess → explore (EDA) → split → train → predict → evaluate, using a single, interpretable **Logistic Regression** model.

---

## 📊 Results

Evaluated on a held-out 20% test set:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8300 |
| Precision | 0.8438 |a
| Recall    | 0.8100 |
| F1-Score  | 0.8265 |

> Numbers above are from a run against a schema-matched synthetic dataset (used when `loan_data.csv` isn't present). Results on the real Kaggle CSV will differ slightly — re-run the script with the real file to update them.

**Confusion Matrix**

|                     | Predicted: Rejected | Predicted: Approved |
|---------------------|:--------------------:|:--------------------:|
| **Actual: Rejected** | 170 (TN) | 30 (FP) |
| **Actual: Approved** | 38 (FN)  | 162 (TP) |

---

## 🗂️ Dataset

| Column | Type | Description |
|---|---|---|
| `person_age` | Numeric | Applicant age |
| `person_gender` | Categorical | Applicant gender |
| `person_education` | Categorical | Highest education level |
| `person_income` | Numeric | Annual income |
| `person_emp_exp` | Numeric | Years of employment experience |
| `person_home_ownership` | Categorical | RENT / OWN / MORTGAGE / OTHER |
| `loan_amnt` | Numeric | Requested loan amount |
| `loan_intent` | Categorical | Purpose of the loan |
| `loan_int_rate` | Numeric | Interest rate offered |
| `loan_percent_income` | Numeric | Loan amount as a fraction of income |
| `cb_person_cred_hist_length` | Numeric | Length of credit history (years) |
| `credit_score` | Numeric | Applicant credit score |
| `previous_loan_defaults_on_file` | Categorical | Yes / No — prior default on record |
| `loan_status` | **Target** | 0 = Rejected, 1 = Approved |

---

## ⚙️ How It Works

1. **Load** — reads `loan_data.csv`. If the file isn't found, a synthetic dataset with the same schema is auto-generated so the pipeline can still be tested end-to-end.
2. **Preprocess**
   - Missing numeric values → filled with the column median
   - Missing categorical values → filled with the column mode
   - Categorical columns → Label Encoded
   - All features → scaled with `StandardScaler`
3. **Explore (EDA)** — saves plots to `plots/`:
   - Target class distribution
   - Feature correlation heatmap
   - Income vs. loan amount (colored by approval)
4. **Split** — 80/20 train/test split, stratified on the target
5. **Train** — `LogisticRegression(max_iter=1000)`
6. **Predict** — generates predictions on the test set
7. **Evaluate** — accuracy, precision, recall, F1-score, confusion matrix, and full classification report; also saves a confusion matrix plot
8. **Bonus** — `predict_loan_approval()`, a small function you can call with a single applicant's raw details to get an instant APPROVED/REJECTED prediction with probability

---

## 🚀 Getting Started

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run it
1. Download `loan_data.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data) (free Kaggle account required).
2. Place `loan_data.csv` in the same folder as the script.
3. Run:
   ```bash
   python loan_approval_classifier.py
   ```
4. Check the console output for metrics, and the `plots/` folder for all visualizations.

> No `loan_data.csv`? The script will still run — it falls back to a synthetic dataset so you can verify the pipeline works before plugging in real data.

---

## 🔮 Making a Prediction

```python
from loan_approval_classifier import predict_loan_approval

predict_loan_approval({
    "person_age": 30,
    "person_gender": "male",
    "person_education": "Bachelor",
    "person_income": 55000,
    "person_emp_exp": 5,
    "person_home_ownership": "RENT",
    "loan_amnt": 10000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.18,
    "cb_person_cred_hist_length": 6,
    "credit_score": 650,
    "previous_loan_defaults_on_file": "No",
})
# -> Prediction: APPROVED  (approval probability: 78.42%)
```

---

## 📁 Project Structure

```
.
├── loan_approval_classifier.py   # main script (load → preprocess → EDA → train → evaluate)
├── loan_data.csv                 # dataset (download from Kaggle, not included in repo)
└── plots/                        # generated on run
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── income_vs_loanamount.png
    └── confusion_matrix_logistic_regression.png
```

---

## 🧠 Key Insight

Standardized logistic regression coefficients show the model leans on **financial-risk signals** rather than demographic ones:

- 🔴 Strongest negative driver: `previous_loan_defaults_on_file`
- 🟢 Strongest positive drivers: `credit_score`, `person_income`
- 🔴 Negative driver: `loan_percent_income` (over-leveraged requests are penalized)
- ⚪ Near-zero effect: `person_gender`, `person_education`, `cb_person_cred_hist_length`

---

## 📌 Notes & Limitations

- Uses a single 80/20 split rather than k-fold cross-validation.
- No hyperparameter tuning (default scikit-learn settings aside from `max_iter`).
- Logistic Regression assumes a linear decision boundary — may underfit non-linear feature interactions compared to tree-based models.
- Any production use of a loan-approval model should include a fairness/bias audit across protected attributes before deployment.

---

## 📄 License

Dataset © original Kaggle uploader — see the [dataset page](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data) for terms. Code in this repo is free to use and modify.
