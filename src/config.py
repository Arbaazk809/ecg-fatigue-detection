"""
config.py

Global configuration for the ECG Fatigue Detection project.
"""

# ==========================
# ECG Configuration
# ==========================

SAMPLING_RATE = 700

LOWCUT = 0.5
HIGHCUT = 40.0

FILTER_ORDER = 4


# ==========================
# Window Configuration
# ==========================

WINDOW_DURATION = 30          # seconds

WINDOW_SIZE = SAMPLING_RATE * WINDOW_DURATION

STEP_SIZE = WINDOW_SIZE // 2  # 50% overlap


# ==========================
# Machine Learning
# ==========================

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ==========================
# File Paths
# ==========================

MODEL_PATH = "models/random_forest.pkl"