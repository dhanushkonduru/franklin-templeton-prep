from sklearn.pipeline import Pipeline

import mlflow

from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

from mlops.mlflow_utils import (

    log_params,

    log_model,

    register_model

)

def train_model(
    X_train,
    y_train
):

    pipeline = Pipeline(

        [

            (
                "model",

                XGBRegressor(

                    n_estimators=100,

                    max_depth=4,

                    learning_rate=0.05,

                    random_state=42

                )

            )

        ]

    )

    pipeline.fit(

        X_train,

        y_train

    )

    log_params(

        {

            "model":"xgboost",

            "n_estimators":100,

            "max_depth":4,

            "learning_rate":0.05

        }

    )

    log_model(

        pipeline,

        "xgboost"

    )
    
    run_id = (

    mlflow.active_run()

    .info

    .run_id

    )

    register_model(

        run_id,

        "xgboost",

        "PortfolioXGBoost"

    )

    return pipeline