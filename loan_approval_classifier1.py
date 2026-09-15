import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Make plots save to files instead of popping up windows (works headless too)
plt.switch_backend("Agg")

RANDOM_STATE = 42


def make_synthetic_dataset(n=2000):
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "person_age": rng.integers(20, 65, n),
        "person_gender": rng.choice(["male", "female"], n),
        "person_education": rng.choice(
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"], n),
        "person_income": rng.integers(15000, 150000, n),
        "person_emp_exp": rng.integers(0, 30, n),
        "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
        "loan_amnt": rng.integers(1000, 35000, n),
        "loan_intent": rng.choice(
            ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE",
             "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], n),
        "loan_int_rate": np.round(rng.uniform(5, 23, n), 2),
        "loan_percent_income": np.round(rng.uniform(0.02, 0.6, n), 2),
        "cb_person_cred_hist_length": rng.integers(2, 30, n),
        "credit_score": rng.integers(400, 850, n),
        "previous_loan_defaults_on_file": rng.choice(["Yes", "No"], n),
    })
    score = (
        (df["credit_score"] - 400) / 450 * 3
        + (df["person_income"] / 150000) * 2
        - df["loan_percent_income"] * 3
        - (df["previous_loan_defaults_on_file"] == "Yes").astype(int) * 2
        + rng.normal(0, 1, n)
    )
    df["loan_status"] = (score > np.median(score)).astype(int)
    return df


DATA_PATH = "loan_data.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded real dataset from '{DATA_PATH}' -> shape: {df.shape}")
else:
    print(f"'{DATA_PATH}' not found in this folder — using a SYNTHETIC "
          f"dataset instead so the pipeline can be tested.\n"
          f"Download the real CSV from Kaggle and place it here to use real data.\n")
    df = make_synthetic_dataset()

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Column info ---")
print(df.info())

print("\n--- Summary statistics ---")
print(df.describe(include="all"))


print("\n--- Missing values per column ---")
print(df.isnull().sum())

# Fill missing numeric values with the median, categorical with the mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object" or str(df[col].dtype) in ("string", "category"):
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

TARGET = "loan_status"

# Identify categorical (text) columns to encode
categorical_cols = [c for c in df.select_dtypes(include=["object", "string", "category"]).columns
                     if c != TARGET]
print(f"\nCategorical columns to encode: {categorical_cols}")

# Simple Label Encoding for each categorical column
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # keep encoders in case we want to decode later

print("\n--- Data after encoding ---")
print(df.head())


os.makedirs("plots", exist_ok=True)

# 3a. Target class balance
plt.figure(figsize=(5, 4))
sns.countplot(x=TARGET, data=df)
plt.title("Loan Status Distribution (0 = Rejected, 1 = Approved)")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.close()

# 3b. Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# 3c. Income vs loan amount, coloured by approval
if "person_income" in df.columns and "loan_amnt" in df.columns:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="person_income", y="loan_amnt", hue=TARGET,
                     data=df, alpha=0.5, palette="Set1")
    plt.title("Income vs Loan Amount by Approval Status")
    plt.tight_layout()
    plt.savefig("plots/income_vs_loanamount.png")
    plt.close()

print("\nSaved EDA plots to the 'plots/' folder "
      "(target_distribution.png, correlation_heatmap.png, income_vs_loanamount.png).")


X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale numeric features (helps Logistic Regression, KNN, SVM converge/perform better)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")



model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n================ Logistic Regression ================")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save a confusion matrix plot
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"])
plt.title("Confusion Matrix — Logistic Regression")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_logistic_regression.png")
plt.close()

best_model = model
best_model_name = "Logistic Regression"


def predict_loan_approval(sample_dict, model=best_model, encoders=encoders,
                           scaler=scaler, feature_order=X.columns):
    """
    Takes a dict of raw feature values (same column names as the original
    dataset), encodes/scales them the same way as training data, and
    returns the model's approval prediction.

    Example:
        predict_loan_approval({
            "person_age": 30, "person_gender": "male",
            "person_education": "Bachelor", "person_income": 55000,
            "person_emp_exp": 5, "person_home_ownership": "RENT",
            "loan_amnt": 10000, "loan_intent": "PERSONAL",
            "loan_int_rate": 11.5, "loan_percent_income": 0.18,
            "cb_person_cred_hist_length": 6, "credit_score": 650,
            "previous_loan_defaults_on_file": "No",
        })
    """
    row = pd.DataFrame([sample_dict])
    for col, le in encoders.items():
        if col in row.columns:
            # handle unseen categories gracefully
            row[col] = row[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else 0
            )
    row = row[feature_order]  # match training column order
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    proba = model.predict_proba(row_scaled)[0][1] if hasattr(model, "predict_proba") else None
    label = "APPROVED" if pred == 1 else "REJECTED"
    if proba is not None:
        print(f"Prediction: {label}  (approval probability: {proba:.2%})")
    else:
        print(f"Prediction: {label}")
    return pred


if __name__ == "__main__":
    print("\n\n================ DEMO PREDICTION ================")
    demo_applicant = X_test.iloc[0].to_dict()
    print("Sample applicant (already-encoded row from test set):", demo_applicant)
    demo_pred = best_model.predict(scaler.transform(pd.DataFrame([demo_applicant])))[0]
    print(f"Predicted: {'APPROVED' if demo_pred == 1 else 'REJECTED'} "
          f"| Actual: {'APPROVED' if y_test.iloc[0] == 1 else 'REJECTED'}")

    print("\nDone! Check the 'plots/' folder for all visualisations, "
          "and edit predict_loan_approval() calls above to test your own applicants.")
