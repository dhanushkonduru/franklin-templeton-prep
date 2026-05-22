def create_features(prices):

    returns = prices.pct_change()

    momentum_3m = prices.pct_change(63)

    momentum_12m = prices.pct_change(252)

    volatility = (
        returns.rolling(63).std()
    )

    features = {
        "returns": returns,
        "momentum_3m": momentum_3m,
        "momentum_12m": momentum_12m,
        "volatility": volatility
    }

    return features