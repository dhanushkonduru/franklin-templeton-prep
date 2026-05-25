import mlflow
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from mlops.mlflow_utils import log_model, log_params


def train_model(X_train, y_train):
    pipeline = Pipeline(
        [
            (
                "model",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                ),
            )
        ]
    )

    pipeline.fit(X_train, y_train)

    log_params(
        {
            "model": "xgboost",
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
        }
    )

    xgb_model = pipeline.named_steps["model"]
    log_model(
        xgb_model,
        artifact_path="model",
        registered_model_name="PortfolioXGBoost",
    )

    if mlflow.active_run() is not None:
        mlflow.log_param("registered_model_name", "PortfolioXGBoost")

    return pipeline