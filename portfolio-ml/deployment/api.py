from fastapi import FastAPI

from pydantic import BaseModel

import mlflow

import pandas as pd


app = FastAPI()


MODEL_NAME = "PortfolioXGBoost"

MODEL_STAGE = "Staging"


model = mlflow.pyfunc.load_model(

    f"models:/{MODEL_NAME}/{MODEL_STAGE}"

)


class StockInput(

    BaseModel

):

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

    return {

        "status":"running"

    }


@app.post(

    "/predict"

)

def predict(

    data: StockInput

):

    features = pd.DataFrame(

        [

            data.dict()

        ]

    )

    prediction = model.predict(

        features

    )[0]

    return {

        "predicted_return":

        float(prediction)

    }