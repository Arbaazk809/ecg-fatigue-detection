from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.predict import predict

app = FastAPI(
    title="ECG Fatigue Detection API",
    description="Random Forest based ECG Fatigue Detection",
    version="1.0.0"
)


class ECGFeatures(BaseModel):
    Mean_RR: float
    Mean_HR: float
    SDNN: float
    RMSSD: float
    Min_RR: float
    Max_RR: float
    Median_RR: float


@app.get("/")
def home():
    return {
        "message": "ECG Fatigue Detection API is running!"
    }


@app.post("/predict")
def predict_fatigue(features: ECGFeatures):

    try:

        df = pd.DataFrame([features.dict()])

        prediction = predict(df)

        fatigue_level = int(prediction[0])

        prediction_text = (
            "Fatigue"
            if fatigue_level == 1
            else "No Fatigue"
        )

        return {
            "fatigue_level": fatigue_level,
            "prediction": prediction_text
        }



    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))