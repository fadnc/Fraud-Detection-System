# src/evaluate.py
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
from preprocessing import load_data, basic_preprocess, train_test_split_df
from utils import plot_confusion_matrix
import shap
import matplotlib.pyplot as plt

def evaluate(model_artifact_path, data_path):
    artifact = joblib.load(model_artifact_path)
    model = artifact['model']
    scaler = artifact['scaler']  # scaler was fit on whole df in basic_preprocess

    # Load & preprocess (consistent with training)
    df = load_data(data_path)
    df_scaled, _ = basic_preprocess(df)  # note: this refits scaler; prefer using saved scaler in real production
    X_train, X_test, y_train, y_test = train_test_split_df(df_scaled, test_size=0.2)

    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    print("Classification report (test):")
    print(classification_report(y_test, y_pred, digits=4))
    roc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC: {roc:.4f}  |  PR-AUC (avg precision): {ap:.4f}")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, savepath=None)

    # SHAP explainability (sample to speed up)
    explainer = shap.TreeExplainer(model)
    # use a small background sample
    background = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(background)

    # Summary plot
    shap.summary_plot(shap_values, background, show=True)
    plt.savefig("artifacts/figures/shap_summary.png", bbox_inches='tight', dpi=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model artifact (joblib)")
    parser.add_argument("--data", type=str, required=True, help="Path to creditcard.csv")
    args = parser.parse_args()
    evaluate(args.model, args.data)
