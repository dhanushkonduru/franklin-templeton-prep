import pandas as pd

def build_dataset(
    features,
    targets,
    fundamentals
):

    dataset = []

    for ticker in targets.columns:

        fundamental_data = fundamentals.get(
            ticker,
            {}
        )

        df = pd.DataFrame({

            "return_1d":
                features["returns"][ticker],

            "momentum_3m":
                features["momentum_3m"][ticker],

            "momentum_12m":
                features["momentum_12m"][ticker],

            "volatility":
                features["volatility"][ticker],

            "pe_ratio":
                fundamental_data.get("pe_ratio"),

            "roe":
                fundamental_data.get("roe"),

            "debt_to_equity":
                fundamental_data.get("debt_to_equity"),

            "revenue_growth":
                fundamental_data.get("revenue_growth"),

            "target":
                targets[ticker]
        })

        df["ticker"] = ticker

        dataset.append(df)

    dataset = pd.concat(dataset)

    dataset = dataset.dropna()

    return dataset