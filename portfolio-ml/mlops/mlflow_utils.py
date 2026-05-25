from mlflow.tracking import MlflowClient

import mlflow
import mlflow.xgboost

client = MlflowClient()

def start_run():

    mlflow.set_experiment(

        "portfolio_ml"

    )

    if mlflow.active_run():

        mlflow.end_run()

    mlflow.start_run()


def end_run():

    if mlflow.active_run():

        mlflow.end_run()


def log_params(params):

    mlflow.log_params(
        params
    )


def log_metrics(

    sharpe,
    drawdown,
    info_ratio

):

    mlflow.log_metric(
        "sharpe",
        sharpe
    )

    mlflow.log_metric(
        "drawdown",
        drawdown
    )

    mlflow.log_metric(
        "information_ratio",
        info_ratio
    )


def log_model(
    model,
    artifact_path,
    registered_model_name=None,
):
    mlflow.xgboost.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )

def register_model(

    run_id,

    artifact_name,

    model_name

):

    model_uri = (

        f"runs:/{run_id}/"

        f"{artifact_name}"

    )

    result = mlflow.register_model(

        model_uri,

        model_name

    )

    return result

def promote_model(

    model_name,

    version,

    stage

):

    client.transition_model_version_stage(

        name=model_name,

        version=version,

        stage=stage

    )