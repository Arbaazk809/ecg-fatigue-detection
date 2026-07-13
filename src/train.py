import numpy as np
import pandas as pd

from src.preprocessing import (
    load_subject,
    extract_ecg,
    extract_labels,
    bandpass_filter,
    detect_r_peaks,
)

from src.hrv_features import (
    compute_rr_intervals,
    extract_windows,
    extract_hrv_features,
)


def get_window_labels(labels, window_size):
    """
    Assign one label to every ECG window using majority voting.
    """

    window_labels = []

    for start in range(0, len(labels) - window_size + 1, window_size):

        window = labels[start:start + window_size]

        values, counts = np.unique(window, return_counts=True)

        majority_label = values[np.argmax(counts)]

        window_labels.append(majority_label)

    return np.array(window_labels)

def create_feature_dataset(subject_path, subject_id):
    """
    Create a feature dataframe for one WESAD subject.
    """

    # Load subject
    data = load_subject(subject_path, subject_id)

    # ECG and labels
    ecg = extract_ecg(data)
    labels = extract_labels(data)

    # Filter ECG
    filtered_ecg = bandpass_filter(ecg)

    # Detect R-peaks
    _, info = detect_r_peaks(filtered_ecg)
    r_peaks = info["ECG_R_Peaks"]

    # RR intervals
    rr_intervals = compute_rr_intervals(r_peaks)

    # HRV windows (30 seconds)
    windows = extract_windows(rr_intervals)

    # HRV feature extraction
    feature_df = extract_hrv_features(windows)

    # Label windows
    label_windows = get_window_labels(labels, window_size=700 * 30)

    # Make lengths equal
    n = min(len(feature_df), len(label_windows))

    feature_df = feature_df.iloc[:n].copy()
    feature_df["Label"] = label_windows[:n]

    return feature_df