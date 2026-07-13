"""
hrv_features.py

Functions for computing Heart Rate Variability (HRV) features
from detected ECG R-peaks.
"""

import numpy as np
import pandas as pd


def compute_rr_intervals(
    r_peaks: np.ndarray,
    sampling_rate: int = 700,
) -> np.ndarray:
    """
    Compute RR intervals from detected R-peaks.

    Parameters
    ----------
    r_peaks : np.ndarray
        Sample positions of detected R-peaks.

    sampling_rate : int, default=700
        ECG sampling frequency (Hz).

    Returns
    -------
    np.ndarray
        RR intervals in milliseconds.
    """

    rr_samples = np.diff(r_peaks)

    rr_intervals = (rr_samples / sampling_rate) * 1000

    return rr_intervals



def calculate_hrv_features(
    rr_intervals: np.ndarray,
) -> dict:
    """
    Compute HRV features for a single RR interval window.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals (milliseconds).

    Returns
    -------
    dict
        Dictionary containing HRV features.
    """

    mean_rr = np.mean(rr_intervals)

    mean_hr = 60000 / mean_rr

    sdnn = np.std(rr_intervals)

    rmssd = np.sqrt(
        np.mean(
            np.diff(rr_intervals) ** 2
        )
    )

    min_rr = np.min(rr_intervals)

    max_rr = np.max(rr_intervals)

    median_rr = np.median(rr_intervals)

    return {
        "Mean_RR": mean_rr,
        "Mean_HR": mean_hr,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "Min_RR": min_rr,
        "Max_RR": max_rr,
        "Median_RR": median_rr,
    }


def extract_hrv_features(
    windows: list,
) -> pd.DataFrame:
    """
    Extract HRV features from multiple RR windows.

    Parameters
    ----------
    windows : list
        List of RR interval windows.

    Returns
    -------
    pandas.DataFrame
        HRV feature table.
    """

    feature_list = []

    for rr_window in windows:

        # Skip windows with too few RR intervals
        if len(rr_window) < 3:
            continue

        features = calculate_hrv_features(rr_window)

        feature_list.append(features)

    return pd.DataFrame(feature_list)