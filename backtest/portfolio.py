import pandas as pd

def construct_portfolio(
    data,
    prediction_column
):

    portfolios = []

    grouped = data.groupby(data.index)

    for date, group in grouped:

        group = group.copy()

        group["rank"] = (
            group[prediction_column]
            .rank(pct=True)
        )

        group["position"] = 0

        group.loc[
            group["rank"] >= 0.9,
            "position"
        ] = 1

        group.loc[
            group["rank"] <= 0.1,
            "position"
        ] = -1

        portfolios.append(group)

    portfolio = pd.concat(portfolios)

    return portfolio