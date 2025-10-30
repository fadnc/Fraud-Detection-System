#  Fraud Detection System

**Project:** Machine Learning pipeline to detect fraudulent credit card transactions (binary classification)  
**Stack:** Python, Pandas, Scikit-learn, XGBoost, Imbalanced-learn, SHAP, Streamlit *(optional)*  
**Dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
Place `creditcard.csv` inside the `data/` folder.

---

##  Project Summary

This repository implements an end-to-end **Fraud Detection** pipeline with the following features:

- **EDA & Preprocessing:** Data exploration, scaling, and imbalance handling  
- **Feature Engineering:** Lightweight, domain-agnostic feature transformations  
- **Model Training & Evaluation:** XGBoost baseline with cross-validation and performance metrics  
- **Model Explainability:** SHAP-based interpretation of model predictions  
- **Deployment (Optional):** Streamlit demo for real-time prediction

The project is modular — you can experiment in notebooks and move to production-ready scripts.

---

##  Quickstart

### 1. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # On Linux/Mac
.venv\Scripts\activate           # On Windows
pip install -r requirements.txt
```

---

### 2. Prepare Data
Download the dataset from Kaggle and place it in the `data/` directory:
```
data/creditcard.csv
```

---

### 3. Train Model
```bash
python src/train.py --data data/creditcard.csv --output artifacts/models/fraud_xgb.pkl
```

---

### 4. Evaluate Model
```bash
python src/evaluate.py --model artifacts/models/fraud_xgb.pkl --data data/creditcard.csv
```

---

### 5. Run Streamlit Demo (Optional)
```bash
streamlit run app/streamlit_app.py
```

---

##  requirements.txt

Below is the list of dependencies required for this project:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.1.1
imbalanced-learn==0.12.3
matplotlib==3.9.2
seaborn==0.13.2
shap==0.45.1
streamlit==1.38.0
joblib==1.4.2
```

Save this as `requirements.txt` in your project root.

---

##  Repository Structure

```
├── data/
│   └── creditcard.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── artifacts/
│   └── models/
├── notebooks/
│   └── eda.ipynb
├── requirements.txt
└── README.md
```

---

##  Model Explainability

- **SHAP values** are used to interpret model predictions.  
- Visual insights reveal which features contribute most to detecting fraud.  
- Helps ensure transparency in model decisions.

---

##  Tech Highlights

-  **XGBoost** for high-performance classification  
-  **SMOTE** for imbalanced data handling  
-  **SHAP** for model interpretability  
-  **Scikit-learn Pipelines** for clean preprocessing  
-  **Streamlit** for user-friendly deployment  

---

##  Future Enhancements

- 🔹 Add Deep Learning models (Autoencoders)  
- 🔹 Deploy API using **FastAPI**  
- 🔹 Integrate **MLflow** for experiment tracking  
- 🔹 Add drift detection for model monitoring  

---

##  Author

**Fadhil Muhammed N C**  
Data Science Enthusiast | Machine Learning | AI | Streamlit Apps  

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
