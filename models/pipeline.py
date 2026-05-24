from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.compose import (
    ColumnTransformer
)

numeric_features = [

    "return_1d",
    "momentum_3m",
    "momentum_12m",
    "volatility",

    "pe_ratio",
    "roe",
    "debt_to_equity",
    "revenue_growth"
]

numeric_transformer = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            numeric_transformer,
            numeric_features
        )
    ]
)