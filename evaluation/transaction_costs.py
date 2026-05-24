def apply_transaction_costs(
    returns,
    turnover,
    cost_per_trade=0.001
):

    costs = turnover * cost_per_trade

    net_returns = returns - costs

    return net_returns