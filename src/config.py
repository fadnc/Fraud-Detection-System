import os

RANDOM_STATE = 42

# Resolve project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
FIG_DIR = os.path.join(ARTIFACT_DIR, "figures")

# Ensure folders exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Model hyperparameters
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "min_child_weight": 1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}


# Data split config
TEST_SIZE = 0.2

# File paths (defaults)
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_xgb_model.pkl")

# Logging and display
VERBOSE = True
