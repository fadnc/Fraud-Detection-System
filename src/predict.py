import argparse
import pandas as pd
import joblib
import numpy as np

def load_artifact(path):
    return joblib.load(path)

def predict_single(artifact_path, input_dict):
    art = load_artifact(artifact_path)
    model = art['model']
    scaler = art['scaler']
    # Build DF with same columns order as training data
    X = pd.DataFrame([input_dict])
    # Use saved scaler to transform Amount and Time (assumes columns exist)
    if 'Time' in X.columns and 'Amount' in X.columns:
        X[['Time','Amount']] = scaler.transform(X[['Time','Amount']])
    probs = model.predict_proba(X)[:,1]
    return probs[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Path to model artifact (joblib)")
    # example features: V1 V2 ... V28 Time Amount
    parser.add_argument("--features", nargs="+", help="Provide feature values in order: V1 V2 ... V28 Time Amount", required=True)
    args = parser.parse_args()

    # For simplicity, expect 30 numeric values (V1..V28, Time, Amount)
    vals = [float(x) for x in args.features]
    cols = [f"V{i}" for i in range(1,29)] + ["Time", "Amount"]
    if len(vals) != len(cols):
        raise SystemExit(f"Expected {len(cols)} feature values, got {len(vals)}")

    input_dict = dict(zip(cols, vals))
    prob = predict_single(args.artifact, input_dict)
    print(f"Predicted fraud probability: {prob:.4f}")
