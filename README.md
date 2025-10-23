# Fraud Detection System

**Project:** Machine learning pipeline to detect fraudulent credit card transactions (binary classification)  
**Stack:** Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn, SHAP, Streamlit (optional)  
**Dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (place `creditcard.csv` in `data/`)

---

## Project Summary

This repository implements an end-to-end fraud detection pipeline:
- EDA and preprocessing (scaling, imbalance handling)
- Feature engineering (light / domain-agnostic)
- Model training and evaluation (XGBoost baseline + cross-validation)
- Model explainability (SHAP)
- Simple Streamlit app for demo/prediction

The repository is organized so you can run experiments from notebooks and productionize using the modular scripts.

---
## Quickstart

1. Create virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
