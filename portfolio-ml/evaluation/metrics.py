import numpy as np

def sharpe_ratio(returns):

    return (
        np.mean(returns)
        /
        np.std(returns)
    ) * np.sqrt(12)

def max_drawdown(cumulative):

    peak = cumulative.cummax()

    drawdown = (
        cumulative - peak
    ) / peak

    return drawdown.min()

def information_ratio(
    strategy_returns,
    benchmark_returns
):

    active_return = (
        strategy_returns
        -
        benchmark_returns
    )

    return (

        active_return.mean()

        /

        active_return.std()

    ) * np.sqrt(12)