
---

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