from evaluation.transaction_costs import (
    apply_transaction_costs
)

def compute_strategy_returns(portfolio):

    portfolio["strategy_return"] = (
        portfolio["position"]
        *
        portfolio["target"]
    )

    returns = portfolio.groupby(
        portfolio.index
    )["strategy_return"].mean()

    turnover = (
        portfolio["position"]
        .diff()
        .abs()
        .fillna(0)
    )

    turnover = turnover.groupby(
        portfolio.index
    ).mean()

    returns = apply_transaction_costs(
        returns,
        turnover
    )

    return returns