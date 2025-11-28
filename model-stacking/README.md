# Model Stacking Heart Disease Classification

Ensemble stacking pipeline combining XGBoost, Random Forest, and Logistic Regression with advanced feature engineering and anti-overfitting strategies for heart disease prediction.

## Objectives

- **End-to-end ML workflow** with simplified feature engineering and rigorous overfitting elimination
- **Multiple approaches tested** with comprehensive comparison (Baseline, Simplified, Diverse+Bagging)
- **Comprehensive evaluation** with 9 metrics and detailed error analysis
- **Best practices** in ensemble diversity, regularization, and data leakage prevention

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
- **Feature Engineering:** 3 cardiovascular-specific features
- **Feature Selection:** Automatic selection of top 18 features
- **Validation:** 5-fold Stratified Cross-Validation

---

## Performance Results

    Metric  Baseline  Simplified  Improvement  Improvement %
 Accuracy  0.902174   0.902174     0.000000           0.00
Precision  0.909091   0.898734     -0.010357         -1.14
   Recall  0.915033   0.928105     0.013072           1.43
 F1-Score  0.912051   0.913158     0.001107           0.12
  ROC-AUC  0.945885   0.945797     -0.000088         -0.01
      MCC  0.801913   0.801805     -0.000108         -0.01

**Top Predictive Features (Simplified - 3 engineered):**
1. Heart_Rate_Reserve - Cardiac capacity indicator
2. Age_Chol_Product - Combined age-cholesterol risk
3. BP_Risk_Score - Normalized blood pressure risk

**Feature Engineering Impact:**  
Simplified approach (3 features) matched baseline performance while maintaining interpretability.

---

## Technical Challenges & Solutions

### Challenge 1: Overfitting (Train-CV Gap > 5%)
**Problem:** Initial models with 20 engineered features showed +5.8% overfitting gap (Train=98.01%, CV=92.19%), indicating memorization rather than generalization.

### Challenge 2: Simplicity vs Complexity Trade-off
**Problem:** Determining optimal feature count - testing 6, 9, and 26 features revealed that simplicity wins on small datasets (918 samples).

---

## Model Comparison: Three Approaches

| Metric     | **Baseline (6 feat)**| **Simplified (9 feat)** |
|------------|----------------------|------------------------ |
| Accuracy   | **90.22%**           | **90.22%**              |
| Precision  | **90.91%**           | 89.87%                  | 
| Recall     | 91.50%               | **92.81%**              | 
| F1-Score   | 91.21%               | **91.32%**              |
| ROC-AUC    | **94.59%**           | **94.58%**              | 
| MCC        | **0.8019**           | **0.8018**              |

### Analysis:

**Key Findings:**
- **Baseline wins** - 6 original features achieved best overall performance (0.9459 ROC-AUC)
- **Simplified matches baseline** - 3 engineered features nearly identical performance (0.9458 ROC-AUC)

---

## Technical Implementation

**Model:** Stacking Classifier
- Base: XGBoost + Random Forest + Logistic Regression
- Meta: Logistic Regression with moderate regularization
- Validation: 5-fold Stratified CV with out-of-fold predictions

**Anti-Overfitting Measures:**
- K-fold stratified cross-validation
- Out-of-fold predictions (OOF)
- L1/L2 regularization
- Sample/Feature subsampling
- Limited tree depth 
- Minimum leaf samples
- passthrough=False to prevent leakage

---

## Key Highlights

 **Excellent Performance** - 90.22% accuracy, 94.59% ROC-AUC   
 **Simplicity Wins** - 6 features outperformed 9 and 26 features  
 **Data Leakage Prevention** - Proper pipelines and OOF predictions

## References

- [Scikit-learn Stacking](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

---

**Educational ML project demonstrating ensemble stacking with systematic overfitting prevention**

