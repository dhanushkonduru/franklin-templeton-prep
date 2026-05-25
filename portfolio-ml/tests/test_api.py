from fastapi.testclient import TestClient
from deployment.api import app

client = TestClient(app)

def test_health():

    response = client.get("/")

    assert response.status_code == 200


def test_predict():

    payload = {

        "return_1d":0.02,

        "momentum_3m":0.15,

        "momentum_12m":0.25,

        "volatility":0.3,

        "pe_ratio":28,

        "roe":0.18,

        "debt_to_equity":0.4,

        "revenue_growth":0.12

    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    prediction = response.json()

    assert "predicted_return" in prediction