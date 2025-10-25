# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE

def load_data(path):
    """Load CSV dataset into DataFrame."""
    df = pd.read_csv(path)
    return df

def basic_preprocess(df):
    """Scale Time and Amount (returns a copy)."""
    df = df.copy()
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    return df, scaler

def train_test_split_df(df, test_size=0.2):
    """Split into train/test (stratified)."""
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

def balance_with_smote(X_train, y_train):
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
