from sklearn.pipeline import Pipeline

from models.pipeline import (
    preprocessor
)

from xgboost import XGBRegressor

def train_model(X_train, y_train):

    model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

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

    model.fit(
        X_train,
        y_train
    )

    return model