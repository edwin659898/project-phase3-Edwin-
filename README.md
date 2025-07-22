# Phase 3 Project: Telecom Churn Classification

**Author:** Edwin Joshua Kiuma  
**School:** Moringa School  
**Date:** July 2025  

---

## Business Understanding

**Problem:** The telecom company wants to predict which customers are likely to churn so they can proactively retain them.  
**Goal:** Build a machine learning model to classify customers as likely to churn (`1`) or not (`0`).  
**Audience:** Customer retention & marketing teams  
**Success Metric:** Prioritize **Recall** — catching true churners is key.

---

## Dataset Summary

- File: `bigml_59c28831336c6604c800002a.csv`
- Customers: ~3,300 rows
- Features: Usage (minutes, calls), Charges, Plans, etc.
- Target: `Churn`

---

## Data Preparation

- Encoded binary columns: Yes/No ➜ 1/0
- Dropped: `Phone`, `Area code`, `State`
- Scaled numeric features using `StandardScaler`
- Train-test split (80/20), stratified

---

## Baseline Model: Logistic Regression

| Metric        | Score |
|---------------|-------|
| Accuracy      | ~86%  |
| Recall        | ~0.38 |
| F1 Score      | ~0.52 |

**Issue:** Many churners were missed (high false negatives)

---

## Tuned Model: Random Forest (with GridSearchCV)

| Metric        | Score |
|---------------|-------|
| Accuracy      | ~94%  |
| Recall        | ~0.76 |
| F1 Score      | ~0.77 |

- Tuned `n_estimators`, `max_depth`, `min_samples_split`
- Significantly better at detecting churners

---

## Recommendations

- Focus retention efforts on customers predicted to churn
- Use the model to proactively trigger retention campaigns
- Consider enriching data (e.g., call center logs, contracts)
- Monitor model drift and retrain as needed

---

## Project Structure

```
├── telecom_churn.ipynb         # Notebook with full analysis
├── Telecom_Churn_Presentation.pptx
├── README.md                   # This file
└── data/
    └── bigml_59c28831336c6604c800002a.csv
```

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn


# Customer Churn Prediction

## Project Overview
This project aims to build a machine learning model to predict whether a customer will churn (leave) or not, using telecom usage data. By identifying customers likely to churn, telecom companies can take proactive retention measures.

## Problem Statement
Customer retention is crucial in the telecom industry. This project addresses the classification problem of predicting customer churn based on service usage and account behavior.

## Dataset
The dataset contains 3,333 customer records with 20+ features, including:

- Account length, call minutes, charges
- Voice mail/international plans
- Customer service calls
- Churn (Target variable: `churn`)

Source: [BigML Telecom Dataset](https://bigml.com/user/gallery/dataset/59c28831336c6604c800002a)

## Data Preparation

- Renamed `Churn?` column to `churn`
- Encoded binary features (Yes/No to 1/0)
- Dropped non-predictive columns: `state`, `area code`, `phone number`
- Scaled features using `StandardScaler`

## ⚙️ Modeling Strategy

| Step        | Approach                              |
|-------------|----------------------------------------|
| Baseline    | Logistic Regression, Decision Tree     |
| Advanced    | Random Forest, Gradient Boosting       |
| Evaluation  | Accuracy, F1, Precision, Recall, Confusion Matrix |

## Model Evaluation

### Final Model: Random Forest

| Metric       | Churn Class (1) |
|--------------|-----------------|
| Precision    | 0.90            |
| Recall       | 0.68            |
| F1-Score     | 0.78            |
| Accuracy     | 0.94            |

Gradient Boosting was a close second with slightly better recall but lower precision.

## Key Insights

- Churners made more customer service calls.
- International plan and day charges had predictive value.
- Class imbalance was moderate (~14% churn rate).

## Conclusion

The Random Forest model effectively predicts churn, enabling early intervention. Future work may include:

- SMOTE or class weighting for imbalance
- Feature selection or PCA
- Deploying with a Flask API

## Requirements

```bash
pandas
scikit-learn
matplotlib
seaborn
