def create_targets(prices):

    future_returns = (
        prices.pct_change(21).shift(-21)
    )

    return future_returns