"""
preprocessing.py

Functions for loading and preprocessing ECG data
from the WESAD dataset.
"""

import os
import pickle
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt

def load_subject(dataset_path: str, subject_id: str) -> dict:
    """
    Load a single subject from the WESAD dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the WESAD dataset.

    subject_id : str
        Example: "S2"

    Returns
    -------
    dict
        Dictionary containing all physiological signals.
    """

    subject_file = os.path.join(
        dataset_path,
        subject_id,
        f"{subject_id}.pkl"
    )

    with open(subject_file, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    return data


def extract_ecg(data: dict) -> np.ndarray:
    """
    Extract chest ECG signal.
    """

    return data["signal"]["chest"]["ECG"]


def extract_labels(data: dict) -> np.ndarray:
    """
    Extract labels.
    """

    return data["label"]

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: int = 700,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to an ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        ECG signal.

    lowcut : float
        Lower cutoff frequency (Hz).

    highcut : float
        Upper cutoff frequency (Hz).

    fs : int
        Sampling frequency.

    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered ECG signal.
    """

    # Convert (N,1) → (N,)
    signal = signal.squeeze()

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def detect_r_peaks(
    signal: np.ndarray,
    sampling_rate: int = 700
):
    """
    Detect R-peaks in an ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        Filtered ECG signal.

    sampling_rate : int
        ECG sampling frequency.

    Returns
    -------
    tuple
        signals : DataFrame
            Processed ECG signals.

        info : dict
            Dictionary containing detected R-peaks.
    """

    signals, info = nk.ecg_process(
        signal,
        sampling_rate=sampling_rate
    )

    return signals, info
