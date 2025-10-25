# src/train.py
import argparse
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config import MODEL_DIR, XGB_PARAMS, RANDOM_STATE
from preprocessing import load_data, basic_preprocess, train_test_split_df, balance_with_smote
from utils import save_model

def train_pipeline(data_path, out_model_path=None, cv_folds=5):
    df = load_data(data_path)
    df_scaled, scaler = basic_preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split_df(df_scaled, test_size=0.2)

    # Option A: balance entire training set with SMOTE (simple)
    X_res, y_res = balance_with_smote(X_train, y_train)

    model = XGBClassifier(**XGB_PARAMS)
    # Quick cross-validation on resampled set (be careful: SMOTE was applied globally here; for production use, do CV with SMOTE inside folds)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for tr_idx, val_idx in skf.split(X_res, y_res):
        Xtr, Xval = X_res.iloc[tr_idx], X_res.iloc[val_idx]
        ytr, yval = y_res.iloc[tr_idx], y_res.iloc[val_idx]
        model.fit(Xtr, ytr)
        yprob = model.predict_proba(Xval)[:,1]
        score = roc_auc_score(yval, yprob)
        cv_scores.append(score)
    mean_cv_auc = np.mean(cv_scores)
    print(f"CV ROC-AUC (on resampled data): {mean_cv_auc:.4f}")

    # Final train on the resampled training data
    model.fit(X_res, y_res)

    if out_model_path is None:
        out_model_path = f"{MODEL_DIR}/fraud_xgb_model.pkl"
    save_model({'model': model, 'scaler': scaler}, out_model_path)
    print(f"Saved model + scaler to {out_model_path}")
    return out_model_path, mean_cv_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", type=str, required=True, help="Path to creditcard.csv")
    parser.add_argument("--output", type=str, default=None, help="Path to save model artifact")
    args = parser.parse_args()

    train_pipeline(args.data, args.output)
