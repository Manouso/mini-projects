from fastapi import FastAPI, Query
from pydantic import BaseModel

import joblib
import os
import pandas as pd

# Ensure feature_engineering is available in __main__ for pickle resolution
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model-voting/src'))) # Adjust the path as necessary
from preprocessing import feature_engineering
sys.modules['__main__'].feature_engineering = feature_engineering # Make sure the module is accessible during unpickling

app = FastAPI()

MODEL_PATH = "model-voting/notebooks/models/voting_bayesian_20251207_220624.pkl"

class InputFeatures(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.post("/predict")
def predict(
    features: InputFeatures,
    model_path: str = Query(default=MODEL_PATH, description="Path to the model file")
):
    # Convert input to DataFrame for model
    features_df = pd.DataFrame([features.dict()])

    # Load model
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}
    model = joblib.load(model_path)

    # Predict
    try:
        prediction = model.predict(features_df)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)
            return {"prediction": int(prediction[0]), "probability": proba[0].tolist()}
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}