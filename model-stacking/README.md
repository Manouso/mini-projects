# Model Stacking Heart Disease Classification

Ensemble stacking pipeline combining XGBoost, Random Forest, and Logistic Regression with advanced feature engineering and anti-overfitting strategies for heart disease prediction.

## Objectives

- **End-to-end ML workflow** with feature engineering and rigorous overfitting prevention
- **Advanced hyperparameter tuning** using RandomizedSearchCV (30 iterations, 10-fold CV)
- **Comprehensive evaluation** with 6 metrics and feature importance analysis
- **Model interpretability** through feature selection and importance analysis
- **Best practices** in stratified sampling, regularization, and feature selection

## Dataset

**Heart Disease UCI** - 918 patients, 11 clinical features
- Binary classification: Heart disease presence (Yes/No)
- Features: Age, Sex, Chest Pain, BP, Cholesterol, ECG, Heart Rate, etc.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook model_stacking.ipynb

# Run all cells sequentially
```

## Project Structure

- **model-stacking/**
  - `model_stacking.ipynb` - Complete implementation
  - `requirements.txt` - Dependencies  
  - `README.md` - Documentation

---

## Pipeline Overview

**Stacking Architecture:**
- **Base Learners:** XGBoost, Random Forest, Logistic Regression
- **Meta-Learner:** Logistic Regression (combines base predictions)
- **Feature Engineering:** 20 cardiovascular-specific features
- **Feature Selection:** Automatic selection of top 18 features (from 35 total)
- **Validation:** 10-fold Stratified Cross-Validation

---

## Performance Results

    Metric  Baseline  Optimized  Improvement  Improvement %
 Accuracy  0.885870   0.891304     0.005435           0.61
Precision  0.893204   0.901961     0.008757           0.98
   Recall  0.901961   0.901961     0.000000           0.00
 F1-Score  0.897561   0.901961     0.004400           0.49
  ROC-AUC  0.931978   0.936902     0.004924           0.53
      MCC  0.768807   0.780010     0.011203           1.46

**Top Predictive Features (from 18 selected):**
1. ST_Slope_Up (21.5% importance)
2. ExerciseAngina (8.4%)
3. ST_Slope_Flat (8.0%)
4. Oldpeak_Squared (3.6%) - Engineered
5. Chol_Risk_Score (3.2%) - Engineered

**Feature Engineering Impact:**  
11 out of 20 top features are engineered (55%), demonstrating the value of domain-specific features.

---

## Technical Challenges & Solutions

### Challenge 1: Overfitting (Train-Test Gap)
**Problem:** Initial models showed train scores significantly higher than test scores (gap > 5%), indicating memorization rather than generalization.

### Challenge 2: Feature Engineering Complexity
**Problem:** Creating meaningful cardiovascular features required domain knowledge while avoiding redundant/correlated features that could increase overfitting.

---

## Model Comparison: Stacking vs LightGBM

| Metric     | **Model Stacking** | **LightGBM** | **Winner** |
|------------|-------------------:|-------------:|------------|
| Accuracy   | **89.13%**        | **89.13%**   | **Tie** ⚖️  |
| Precision  | **90.20%**        | **90.20%**   | **Tie** ⚖️  |
| Recall     | **90.20%**        | **90.20%**   | **Tie** ⚖️  |
| F1-Score   | **90.20%**        | **90.20%**   | **Tie** ⚖️  |
| ROC-AUC    | **93.69%**        | 92.96%       | Stacking 🏆 |
| MCC        | **78.00%**        | **78.00%**   | **Tie** ⚖️  |

### Analysis:

**Performance Parity:**
- 🎯 **Identical classification metrics** - Both models achieve exactly 89.13% accuracy and 90.20% precision/recall/F1
- 📊 **Stacking has slightly better ROC-AUC** (+0.73%) - better probability calibration
- ⚖️ **Equal MCC scores** - Both models have identical balance between precision and recall

---

## Technical Implementation

**Model:** Stacking Classifier
- Base: XGBoost + Random Forest + Logistic Regression
- Meta: Logistic Regression with strong regularization
- Feature Selection: SelectKBest (f_classif scoring)

**Anti-Overfitting Measures:**
- K-fold stratified cross-validation
- Feature selection
- L1/L2 regularization
- Data/Feature sampling
- Limited tree depth 
- Increased minimum leaf samples

---

## Key Highlights

✅ **Excellent Performance** - 89.13% accuracy, 93.69% ROC-AUC  
✅ **Rigorous Overfitting Control** - Gap reduced to 4.02%  
✅ **Feature Selection** - 18 most predictive features identified  
✅ **Performance Parity with LightGBM** - Identical classification metrics

## References

- [Scikit-learn Stacking](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

---

**Educational ML project demonstrating ensemble stacking with systematic overfitting prevention**

