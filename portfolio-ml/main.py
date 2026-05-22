from data.data_loader import load_price_data
from data.fundamentals import fetch_fundamentals

from features.feature_engineering import create_features
from features.targets import create_targets
from features.build_dataset import build_dataset

from evaluation.backtest_engine import walk_forward_backtest

from backtest.portfolio import construct_portfolio

from evaluation.performance import compute_strategy_returns

from evaluation.metrics import (
    sharpe_ratio,
    max_drawdown,
    information_ratio
)

from models.regime_detection import detect_regimes

FEATURE_COLUMNS = [
    "return_1d",
    "momentum_3m",
    "momentum_12m",
    "volatility",
    "pe_ratio",
    "roe",
    "debt_to_equity",
    "revenue_growth"
]

tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "UNH",
    "TSLA"
]

prices = load_price_data(
    tickers,
    "2015-01-01",
    "2024-01-01"
)

fundamentals = fetch_fundamentals(
    tickers
)

features = create_features(prices)

targets = create_targets(prices)

dataset = build_dataset(
    features,
    targets,
    fundamentals
)

regime_features = dataset[
    [
        "volatility",
        "momentum_12m"
    ]
]

dataset["regime"] = detect_regimes(
    regime_features
)

results = walk_forward_backtest(
    dataset,
    FEATURE_COLUMNS
)

xgb_portfolio = construct_portfolio(
    results,
    "xgb_prediction"
)

mlp_portfolio = construct_portfolio(
    results,
    "mlp_prediction"
)

xgb_returns = compute_strategy_returns(
    xgb_portfolio
)

mlp_returns = compute_strategy_returns(
    mlp_portfolio
)

xgb_sharpe = sharpe_ratio(
    xgb_returns
)

mlp_sharpe = sharpe_ratio(
    mlp_returns
)

xgb_drawdown = max_drawdown(
    (1 + xgb_returns).cumprod()
)

mlp_drawdown = max_drawdown(
    (1 + mlp_returns).cumprod()
)

benchmark = prices.pct_change().mean(axis=1)

xgb_ir = information_ratio(
    xgb_returns,
    benchmark.reindex(xgb_returns.index).fillna(0)
)

mlp_ir = information_ratio(
    mlp_returns,
    benchmark.reindex(mlp_returns.index).fillna(0)
)

print("\nFINAL MODEL COMPARISON")

print("\nXGBOOST")
print(f"Sharpe Ratio: {xgb_sharpe:.2f}")
print(f"Max Drawdown: {xgb_drawdown:.2%}")
print(f"Information Ratio: {xgb_ir:.2f}")

print("\nMLP")
print(f"Sharpe Ratio: {mlp_sharpe:.2f}")
print(f"Max Drawdown: {mlp_drawdown:.2%}")
print(f"Information Ratio: {mlp_ir:.2f}")