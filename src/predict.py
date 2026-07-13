import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model():
    """
    Load the trained Random Forest model.
    """
    return joblib.load(MODEL_PATH)


def predict(features: pd.DataFrame):
    """
    Predict fatigue class from HRV features.
    """
    model = load_model()
    return model.predict(features)


def predict_proba(features: pd.DataFrame):
    """
    Return prediction probabilities.
    """
    model = load_model()
    return model.predict_proba(features)