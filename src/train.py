"""
train.py

Functions for training machine learning models
for ECG-Based Fatigue Detection.
"""

import joblib
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
)


def load_feature_dataset(feature_df: pd.DataFrame):
    """
    Split the feature dataframe into features (X)
    and labels (y).
    """

    X = feature_df.drop(columns=["Label"])
    y = feature_df["Label"]

    return X, y


def split_dataset(X, y):
    """
    Split the dataset into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, model_path=MODEL_PATH):
    """
    Save the trained model.
    """

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")