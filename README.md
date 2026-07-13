# ECG-Based Fatigue Detection using Machine Learning

A machine learning project for detecting fatigue from Electrocardiogram (ECG) signals using Heart Rate Variability (HRV) features and a Random Forest classifier. The project includes ECG preprocessing, HRV feature extraction, model training, evaluation, and deployment as a REST API using FastAPI and Docker.

---

## Project Overview

Fatigue can negatively affect human performance, concentration, and safety. Physiological signals such as Electrocardiogram (ECG) provide valuable information about the autonomic nervous system, making them useful for fatigue detection.

This project processes raw ECG recordings, extracts Heart Rate Variability (HRV) features, trains a Random Forest classifier, and exposes the trained model through a FastAPI REST API. The application is fully containerized using Docker, making it easy to deploy and reproduce.

---

## Features

- ECG signal preprocessing
- Butterworth band-pass filtering
- R-peak detection
- RR interval computation
- HRV feature extraction
- Random Forest classifier
- Model evaluation
- FastAPI REST API
- Interactive Swagger documentation
- Docker containerization
