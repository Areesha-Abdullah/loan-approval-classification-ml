
# Loan Approval Prediction (Imbalanced Classification)

This project implements a complete machine learning pipeline for predicting loan approval decisions using applicant financial and demographic data. The task focuses on binary classification under class imbalance, emphasizing the importance of evaluation metrics beyond accuracy.

The repository includes:
- Logistic Regression (from scratch)
- Logistic Regression (scikit-learn)
- Decision Tree classifier
- Imbalance handling using SMOTE
- Comparative evaluation using precision, recall, F1-score, and confusion matrices

---

## 📌 Problem Statement

Given applicant information such as income, credit score, education level, and asset values, the goal is to predict whether a loan application should be:

- **Approved (1)**
- **Rejected (0)**

This mirrors real-world credit risk assessment problems commonly faced by banks and fintech organizations.

---

## 📂 Dataset Overview

Typical features include:
- Education level
- Self-employment status
- Annual income
- Loan amount and loan term
- CIBIL (credit) score
- Asset values (residential, commercial, luxury, bank)
- Target variable: `loan_status` (Approved / Rejected)

The dataset is **imbalanced**, with more approved loans than rejected ones.

---

## 🧠 ML Concepts Demonstrated

- Supervised learning (binary classification)
- Handling imbalanced datasets
- Gradient descent–based optimization
- Feature scaling for stable convergence
- Model comparison and evaluation
- Business-relevant performance metrics

---

## 🔧 Project Structure
# Loan Approval Prediction (Imbalanced Classification)

This project implements a complete machine learning pipeline for predicting loan approval decisions using applicant financial and demographic data. The task focuses on binary classification under class imbalance, emphasizing the importance of evaluation metrics beyond accuracy.

The repository includes:
- Logistic Regression (from scratch)
- Logistic Regression (scikit-learn)
- Decision Tree classifier
- Imbalance handling using SMOTE
- Comparative evaluation using precision, recall, F1-score, and confusion matrices

---

## 📌 Problem Statement

Given applicant information such as income, credit score, education level, and asset values, the goal is to predict whether a loan application should be:

- **Approved (1)**
- **Rejected (0)**

This mirrors real-world credit risk assessment problems commonly faced by banks and fintech organizations.

---

## 📂 Dataset Overview

Typical features include:
- Education level
- Self-employment status
- Annual income
- Loan amount and loan term
- CIBIL (credit) score
- Asset values (residential, commercial, luxury, bank)
- Target variable: `loan_status` (Approved / Rejected)

The dataset is **imbalanced**, with more approved loans than rejected ones.

---

## 🧠 ML Concepts Demonstrated

- Supervised learning (binary classification)
- Handling imbalanced datasets
- Gradient descent–based optimization
- Feature scaling for stable convergence
- Model comparison and evaluation
- Business-relevant performance metrics

---

## 🔧 Project Structure

TASK_03/
│
├── src/
│ ├── preprocessing.py
│ ├── train.py
│ ├── logisticReg_scratch.py
│ └── evaluation.py
│
├── run.py
├── README.md
├── requirements.txt
└── .gitignore

## 🔄 Methodology

### 1️⃣ Data Preprocessing
- Missing value handling (median for numerical, mode for categorical)
- Encoding categorical features
- Feature scaling (StandardScaler) for Logistic Regression
- Stratified train-test split to preserve class distribution

---

### 2️⃣ Models Implemented

#### 🔹 Logistic Regression (From Scratch)
- Implemented using NumPy
- Gradient descent optimization
- Sigmoid activation with numerical stability
- Demonstrates low-level understanding of model mechanics

#### 🔹 Logistic Regression (scikit-learn)
- Industry-standard baseline
- Used for performance comparison

#### 🔹 Decision Tree
- Captures non-linear relationships
- Easy to interpret
- Less sensitive to feature scaling

---

### 3️⃣ Handling Class Imbalance
- SMOTE (Synthetic Minority Over-sampling Technique) applied **only on training data**
- Improves recall for minority (rejected loan) class
- Prevents biased learning toward majority class

---

## 📊 Evaluation Metrics

Models are evaluated using:
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Accuracy alone is avoided due to class imbalance.

---

## 📈 Results Summary

### 🔹 Logistic Regression (From Scratch)
- F1-score: **0.93**
- Accuracy: **0.91**
- Balanced precision and recall across classes

### 🔹 Logistic Regression (scikit-learn)
- F1-score: **0.93**
- Accuracy: **0.91**
- Slightly more stable than scratch implementation

### 🔹 Decision Tree
- F1-score: **0.97**
- Accuracy: **0.97**
- Strong performance due to non-linear feature interactions

### 🔹 Logistic Regression + SMOTE
- Improved balance between approval and rejection recall
- F1-score: **0.93**

### 🔹 Decision Tree + SMOTE
- Best overall performance
- F1-score: **0.97**
- High recall for both classes

---

## 🏁 Key Takeaways

- Accuracy can be misleading for imbalanced datasets
- Precision and recall provide better insight into real-world risk
- SMOTE improves minority-class detection
- Decision Trees outperform linear models on non-linear data
- From-scratch implementation reinforces core ML fundamentals

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python run.py