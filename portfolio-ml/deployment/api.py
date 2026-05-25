from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mlflow
import pandas as pd


app = FastAPI()


MODEL_NAME = "PortfolioXGBoost"
MODEL_STAGE = "Production"


try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception:
    model = None


class StockInput(BaseModel):
    return_1d: float
    momentum_3m: float
    momentum_12m: float
    volatility: float
    pe_ratio: float
    roe: float
    debt_to_equity: float
    revenue_growth: float


@app.get("/")
def health():
    return {"status": "running", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: StockInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    features = pd.DataFrame([data.model_dump()])
    prediction = model.predict(features)[0]
    return {"predicted_return": float(prediction)}