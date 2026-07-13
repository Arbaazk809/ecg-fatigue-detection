"""
evaluate.py

Functions for evaluating machine learning models
for ECG-Based Fatigue Detection.
"""

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.

    Parameters
    ----------
    model : sklearn estimator
        Trained machine learning model.

    X_test : pandas.DataFrame
        Test features.

    y_test : pandas.Series
        True labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics


def print_classification_report(model, X_test, y_test):
    """
    Print classification report.
    """

    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix.
    """

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()