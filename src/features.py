# src/features.py
import numpy as np
import pandas as pd

def add_amount_bin(X, bins=[0, 1, 10, 50, 100, 500, 1000]):
    """Create a categorical bin of transaction amount"""
    X = X.copy()
    X['amount_bin'] = pd.cut(X['Amount'], bins=bins, labels=False, include_lowest=True)
    # one-hot encode
    return pd.get_dummies(X, columns=['amount_bin'], prefix='amtbin', drop_first=True)

def add_time_of_day(X):
    """Assumes Time is scaled; we can't easily get real hour without original unscaled values.
       This function is a placeholder if you store raw time. For current dataset we skip.
    """
    return X
