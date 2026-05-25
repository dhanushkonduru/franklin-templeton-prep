from mlops.mlflow_utils import (

    promote_model

)

promote_model(

    "PortfolioXGBoost",

    1,

    "Staging"

)

print(

    "Promoted"

)