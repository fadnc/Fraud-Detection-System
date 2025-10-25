# src/utils.py
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def plot_confusion_matrix(y_true, y_pred, labels=[0,1], savepath=None):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues')
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
    else:
        plt.show()

def plot_heatmap(df, figsize=(10,8), savepath=None):
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
    else:
        plt.show()
