"""
dataset.py

Functions for creating machine learning datasets
from ECG recordings.
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    bandpass_filter,
    detect_r_peaks,
)

from src.hrv_features import (
    compute_rr_intervals,
    calculate_hrv_features,
)


def create_windows(
    signal: np.ndarray,
    window_size: int,
    step_size: int,
) -> list:
    """
    Split a signal into overlapping windows.
    """

    windows = []

    for start in range(
        0,
        len(signal) - window_size + 1,
        step_size,
    ):

        end = start + window_size

        windows.append(signal[start:end])

    return windows


def assign_window_labels(
    labels: np.ndarray,
    window_size: int,
    step_size: int,
) -> list:
    """
    Assign one label to each ECG window
    using majority voting.
    """

    window_labels = []

    for start in range(
        0,
        len(labels) - window_size + 1,
        step_size,
    ):

        end = start + window_size

        window = labels[start:end]

        values, counts = np.unique(
            window,
            return_counts=True,
        )

        majority_label = values[np.argmax(counts)]

        window_labels.append(majority_label)

    return window_labels


def build_feature_dataset(
    windows: list,
    labels: list,
    sampling_rate: int = 700,
) -> pd.DataFrame:
    """
    Convert ECG windows into HRV feature vectors.
    """

    feature_rows = []

    for window, label in zip(windows, labels):

        try:

            filtered = bandpass_filter(window)

            _, info = detect_r_peaks(
                filtered,
                sampling_rate=sampling_rate,
            )

            r_peaks = info["ECG_R_Peaks"]

            if len(r_peaks) < 5:
                continue

            rr = compute_rr_intervals(
                r_peaks,
                sampling_rate=sampling_rate,
            )

            features = calculate_hrv_features(rr)

            features["Label"] = label

            feature_rows.append(features)

        except Exception as e:
            print(f"Skipping windows: {e}")

    return pd.DataFrame(feature_rows)