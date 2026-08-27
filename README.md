# 📡 Telecom Customer Churn Prediction

## 📌 Project Overview

Customer churn is a major challenge for telecom companies. Losing customers can be costly, especially when the company could have identified at-risk customers early and taken preventive action.

This project builds a machine learning solution to predict which telecom customers are likely to churn.

Two classification models were developed and compared:

- 🌲 Random Forest
- 🚀 XGBoost

The models were trained using the same dataset, preprocessing pipeline, feature selection approach, and train/test split to ensure a fair comparison.

---

## 🎯 Business Objective

The main goal is to help the telecom company's retention team identify customers who are at high risk of churn.

The model can help the company:

- Identify customers likely to leave
- Prioritize retention efforts
- Take action before customers churn
- Understand the main factors associated with churn
- Make data-driven retention decisions

Because missing a potential churner can be costly, the project focuses on metrics such as **Recall and F1-score**, rather than relying on Accuracy alone.

---

## 📊 Dataset

The dataset contains approximately **80,000 customer records** and **24 original columns**.

The features cover several areas:

### Customer Demographics
- Age
- Gender
- Senior Citizen
- Partner
- Dependents

### Account & Contract Information
- Tenure Months
- Contract Type
- Payment Method
- Paperless Billing

### Services
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Tech Support
- Streaming TV
- Streaming Movies

### Usage & Billing
- Average Monthly GB Usage
- Number of Support Calls
- Monthly Charges
- Total Charges

### Target
- Churn

---

## 🔍 Data Exploration & Cleaning

The dataset was explored to identify:

- Missing values
- Invalid numerical values
- Unusual outliers
- Duplicate or suspicious features
- Identifier columns
- Target class distribution
- Relationships between features and churn

Examples of identified issues included unrealistic values in the `Age` column and missing values in usage and billing-related variables.

Cleaning decisions were made based on the meaning of each feature rather than applying the same treatment to every column.

---

## 🛠️ Feature Engineering

Additional features were created to help the models capture useful customer behavior.

One of the engineered features was:

### Number_of_Services

This feature represents the number of services used by each customer.

---

## 🎯 Feature Selection

After removing features that did not provide useful predictive information, **SelectKBest with ANOVA F-test (`f_classif`)** was used to select the most relevant features.

The feature selection step was included inside the machine learning pipeline to prevent information leakage.

---

## 🤖 Machine Learning Models

### Random Forest

A Random Forest classifier was trained and hyperparameter-tuned.

Best parameters:

- Number of estimators: 300
- Maximum depth: 15
- Minimum samples split: 2
- Minimum samples leaf: 2
- Selected features: 30

### XGBoost

An XGBoost classifier was also trained and tuned.

Best parameters:

- Number of estimators: 500
- Maximum depth: 3
- Learning rate: 0.03
- Subsample: 1.0
- Column sampling: 1.0
- Selected features: 25

---

## 📈 Model Performance

Both models were evaluated on the **same test set**.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 73.84% | **74.33%** |
| Precision | 61.27% | **62.31%** |
| Recall | 40.23% | **41.43%** |
| F1 Score | 48.57% | **49.77%** |
| ROC-AUC | 76.23% | **77.08%** |

### 🏆 Best Model

**XGBoost performed slightly better than Random Forest across all evaluated metrics.**

However, the default classification threshold of `0.50` resulted in relatively low recall.

Since the business goal is to identify customers at risk of churn, additional threshold analysis was performed.

---

## 🎚️ Threshold Optimization

Different classification thresholds were tested for XGBoost.

The threshold of **0.30** produced the best F1-score among the tested thresholds.

At a threshold of `0.30`:

- Precision: **49.82%**
- Recall: **72.25%**
- F1 Score: **58.98%**

Compared with the default threshold of `0.50`:

- Recall increased from **41.43% → 72.25%**
- F1 Score increased from **49.77% → 58.98%**

This makes the model more suitable for proactive customer retention.

---

## 🔎 Feature Importance

The two models showed some agreement but also important differences.

### Random Forest

Random Forest placed high importance on:

- Total Charges
- Monthly Charges
- Tenure
- Average Monthly GB Usage
- Month-to-month Contract
- Number of Support Calls

### XGBoost

XGBoost placed high importance on:

- Internet Service
- Month-to-month Contract
- Fiber Optic Internet
- Tech Support
- Number of Support Calls
- Tenure

Both models identified **contract type, tenure, and support-related variables** as important factors, while XGBoost placed substantially more emphasis on internet service.

Feature importance indicates how much the models relied on features for prediction and should not be interpreted as direct causal relationships.

---

## 🚀 Deployment

The final XGBoost pipeline was saved using Joblib and deployed using **Streamlit**.

The application allows users to enter customer information and receive:

- Churn probability
- Risk level
- Churn prediction
- Retention recommendation

### Production Decision

**XGBoost is recommended as the production model.**

A threshold of **0.30** is recommended when the business prioritizes identifying more potential churners and is willing to accept additional false positives.

The final threshold should ultimately be adjusted according to the company's retention budget and the relative cost of false positives versus missed churners.

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib
- Streamlit
- Jupyter Notebook

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── customer_churn_xgb_pipeline.pkl
├── requirements.txt
├── README.md
└── notebook/
    └── customer_churn_prediction.ipynb
