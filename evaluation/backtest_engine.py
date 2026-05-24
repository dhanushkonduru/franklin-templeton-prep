import pandas as pd

from models.train_pipeline import (
    train_model
)

from models.train_mlp import (
    train_mlp
)

from models.predict_mlp import (
    predict_mlp
)

from mlops.mlflow_utils import (

    start_run,

    log_metrics,

    end_run

)

from evaluation.metrics import (

    sharpe_ratio,

    max_drawdown,

    information_ratio

)


def walk_forward_backtest(

    dataset,

    feature_columns

):
    start_run()

    dataset = dataset.sort_index()

    years = sorted(

        dataset.index.year.unique()

    )

    predictions = []

    last_xgb_model = None

    for i in range(

        5,

        len(years)-1

    ):

        train_years = years[:i]

        test_year = years[i]

        train_data = dataset[

            dataset.index.year.isin(

                train_years

            )

        ]

        test_data = dataset[

            dataset.index.year == test_year

        ]

        X_train = train_data[
            feature_columns
        ]

        y_train = train_data[
            "target"
        ]

        X_test = test_data[
            feature_columns
        ]

        xgb_model = train_model(

            X_train,

            y_train

        )

        last_xgb_model = xgb_model

        xgb_preds = xgb_model.predict(

            X_test

        )

        mlp_model = train_mlp(

            X_train.fillna(0),

            y_train

        )

        mlp_preds = predict_mlp(

            mlp_model,

            X_test.fillna(0)

        )

        test_data = test_data.copy()

        test_data[
            "xgb_prediction"
        ] = xgb_preds

        test_data[
            "mlp_prediction"
        ] = mlp_preds

        predictions.append(

            test_data

        )

        print(

            f"Completed Walk-Forward Test: {test_year}",

            flush=True

        )

    final_predictions = pd.concat(

        predictions

    )

    strategy_returns = final_predictions[
    "xgb_prediction"
    ]
    
    benchmark_returns = final_predictions[
        "target"
    ]

    sharpe = sharpe_ratio(

        strategy_returns

    )

    drawdown = max_drawdown(

        strategy_returns

    )

    info_ratio = information_ratio(

        strategy_returns,

        benchmark_returns

    )

    log_metrics(

        sharpe,

        drawdown,

        info_ratio

    )

    end_run()

    return (

        final_predictions,

        last_xgb_model

    )