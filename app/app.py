import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

@st.cache_resource
def load_artifact(path):
    return joblib.load(path)

MODEL_PATH = Path(__file__).parents[1] / "artifacts" / "models" / "fraud_xgb_model.pkl"

st.title("Fraud Detection Demo")
st.write("Enter transaction features (V1..V28, Time, Amount) or upload a CSV with these columns.")

artifact = load_artifact(str(MODEL_PATH))
model = artifact['model']
scaler = artifact['scaler']

# Option: let user upload CSV
uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])
    # scale Time and Amount using saved scaler
    if {'Time', 'Amount'}.issubset(df.columns):
        df[['Time','Amount']] = scaler.transform(df[['Time','Amount']])
    probs = model.predict_proba(df)[:,1]
    df['fraud_prob'] = probs
    st.dataframe(df.head())
else:
    st.subheader("Manual input")
    cols = []
    inputs = {}
    for i in range(1,29):
        val = st.number_input(f"V{i}", value=0.0, format="%.6f")
        inputs[f"V{i}"] = val
    time_val = st.number_input("Time", value=0.0)
    amount_val = st.number_input("Amount", value=0.0)
    inputs['Time'] = time_val
    inputs['Amount'] = amount_val

    if st.button("Predict"):
        X = pd.DataFrame([inputs])
        X[['Time','Amount']] = scaler.transform(X[['Time','Amount']])
        prob = model.predict_proba(X)[:,1][0]
        st.write(f"Fraud probability: **{prob:.4f}**")
