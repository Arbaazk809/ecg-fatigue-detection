import numpy as np
import pandas as pd

def extract_hrv_features(windows):
    """
    Extract HRV features from multiple RR windows.
    """

    all_features = []

    for rr_intervals in windows:

        mean_rr = np.mean(rr_intervals)

        mean_hr = 60000 / mean_rr

        sdnn = np.std(rr_intervals)

        rmssd = np.sqrt(
            np.mean(np.diff(rr_intervals) ** 2)
        )

        min_rr = np.min(rr_intervals)

        max_rr = np.max(rr_intervals)

        median_rr = np.median(rr_intervals)

        features = {
            "Mean_RR": mean_rr,
            "Mean_HR": mean_hr,
            "SDNN": sdnn,
            "RMSSD": rmssd,
            "Min_RR": min_rr,
            "Max_RR": max_rr,
            "Median_RR": median_rr
        }

        all_features.append(features)

    return pd.DataFrame(all_features)