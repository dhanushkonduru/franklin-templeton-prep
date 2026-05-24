import os

# Avoid OpenMP deadlocks when XGBoost/sklearn/torch run together.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from backtest.portfolio import construct_portfolio
from data.data_loader import load_price_data
from data.fundamentals import fetch_fundamentals
from evaluation.backtest_engine import walk_forward_backtest
from evaluation.metrics import (
    information_ratio,
    max_drawdown,
    sharpe_ratio,
)
from evaluation.performance import compute_strategy_returns
from features.build_dataset import build_dataset
from features.feature_engineering import create_features
from features.targets import create_targets
from models.regime_detection import detect_regimes


FEATURE_COLUMNS = [
    "return_1d",
    "momentum_3m",
    "momentum_12m",
    "volatility",
    "pe_ratio",
    "roe",
    "debt_to_equity",
    "revenue_growth",
]

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "UNH",
    "TSLA",
]

OUTPUT_DIR = Path("output")


def log(message):

    print(message, flush=True)


def save_predictions(results, path):

    export = results.assign(date=results.index).reset_index(drop=True)

    columns = [
        "date",
        "ticker",
        "target",
        "xgb_prediction",
        "mlp_prediction",
    ]

    export[columns].to_csv(path, index=False)


def save_strategy_returns(xgb_returns, mlp_returns, path):

    frame = pd.DataFrame(
        {
            "date": xgb_returns.index,
            "xgb_return": xgb_returns.values,
            "mlp_return": mlp_returns.reindex(xgb_returns.index).values,
        }
    )

    frame.to_csv(path, index=False)


def save_metrics(xgb_sharpe, mlp_sharpe, xgb_drawdown, mlp_drawdown, xgb_ir, mlp_ir, path):

    payload = {
        "xgboost": {
            "sharpe_ratio": float(xgb_sharpe),
            "max_drawdown": float(xgb_drawdown),
            "information_ratio": float(xgb_ir),
        },
        "mlp": {
            "sharpe_ratio": float(mlp_sharpe),
            "max_drawdown": float(mlp_drawdown),
            "information_ratio": float(mlp_ir),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_cumulative_returns(xgb_returns, mlp_returns, benchmark, path):

    xgb_curve = (1 + xgb_returns).cumprod()
    mlp_curve = (1 + mlp_returns).cumprod()
    bench = benchmark.reindex(xgb_returns.index).fillna(0)
    bench_curve = (1 + bench).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(xgb_curve.index, xgb_curve.values, label="XGBoost")
    plt.plot(mlp_curve.index, mlp_curve.values, label="MLP")
    plt.plot(bench_curve.index, bench_curve.values, label="Benchmark", linestyle="--")
    plt.title("Cumulative Strategy Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_drawdown(xgb_returns, mlp_returns, path):

    xgb_curve = (1 + xgb_returns).cumprod()
    mlp_curve = (1 + mlp_returns).cumprod()

    def drawdown_series(cumulative):

        peak = cumulative.cummax()
        return (cumulative - peak) / peak

    plt.figure(figsize=(10, 6))
    plt.plot(xgb_curve.index, drawdown_series(xgb_curve), label="XGBoost")
    plt.plot(mlp_curve.index, drawdown_series(mlp_curve), label="MLP")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature_importance(xgb_model, feature_columns, path):

    if xgb_model is None:
        return

    model = xgb_model.named_steps["model"]
    importances = model.feature_importances_

    order = sorted(range(len(feature_columns)), key=lambda i: importances[i])

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_columns[i] for i in order],
        [importances[i] for i in order],
    )
    plt.title("XGBoost Feature Importance (last walk-forward fold)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def print_final_comparison(xgb_sharpe, mlp_sharpe, xgb_drawdown, mlp_drawdown, xgb_ir, mlp_ir):

    log("\nFINAL MODEL COMPARISON")

    log("\nXGBOOST")
    log(f"Sharpe Ratio: {xgb_sharpe:.2f}")
    log(f"Max Drawdown: {abs(xgb_drawdown):.2%}")
    log(f"Information Ratio: {xgb_ir:.2f}")

    log("\nMLP")
    log(f"Sharpe Ratio: {mlp_sharpe:.2f}")
    log(f"Max Drawdown: {abs(mlp_drawdown):.2%}")
    log(f"Information Ratio: {mlp_ir:.2f}")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=== Portfolio ML pipeline ===")

    log("\n[1/6] Loading price data...")
    prices = load_price_data(
        TICKERS,
        "2010-01-01",
        "2024-01-01",
    )

    log("\n[2/6] Fetching fundamentals...")
    fundamentals = fetch_fundamentals(TICKERS)

    log("\n[3/6] Engineering features and targets...")
    features = create_features(prices)
    targets = create_targets(prices)

    dataset = build_dataset(
        features,
        targets,
        fundamentals,
    )

    log(f"Dataset ready: {len(dataset)} rows.")

    log("\n[4/6] Detecting market regimes...")
    regime_features = dataset[
        [
            "volatility",
            "momentum_12m",
        ]
    ]

    dataset["regime"] = detect_regimes(regime_features)

    log("\n[5/6] Running walk-forward backtest (may take several minutes)...")
    results, last_xgb_model = walk_forward_backtest(
        dataset,
        FEATURE_COLUMNS,
    )

    log("\n[6/6] Constructing portfolios and computing metrics...")
    xgb_portfolio = construct_portfolio(results, "xgb_prediction")
    mlp_portfolio = construct_portfolio(results, "mlp_prediction")

    xgb_returns = compute_strategy_returns(xgb_portfolio)
    mlp_returns = compute_strategy_returns(mlp_portfolio)

    xgb_sharpe = sharpe_ratio(xgb_returns)
    mlp_sharpe = sharpe_ratio(mlp_returns)

    xgb_drawdown = max_drawdown((1 + xgb_returns).cumprod())
    mlp_drawdown = max_drawdown((1 + mlp_returns).cumprod())

    benchmark = prices.pct_change().mean(axis=1)

    xgb_ir = information_ratio(
        xgb_returns,
        benchmark.reindex(xgb_returns.index).fillna(0),
    )

    mlp_ir = information_ratio(
        mlp_returns,
        benchmark.reindex(mlp_returns.index).fillna(0),
    )
    
    dataset = build_dataset(
    features,
    targets,
    fundamentals,
    )

    log(
        f"Dataset ready: {len(dataset)} rows."
    )

    dataset.to_csv(

        OUTPUT_DIR /

        "final_dataset.csv",

        index=False

    )

    log(

        "Saved dataset for monitoring"

    )

    log("\nSaving artifacts to output/ ...")
    save_predictions(results, OUTPUT_DIR / "predictions.csv")
    xgb_portfolio.to_csv(OUTPUT_DIR / "xgb_portfolio.csv")
    mlp_portfolio.to_csv(OUTPUT_DIR / "mlp_portfolio.csv")
    save_strategy_returns(
        xgb_returns,
        mlp_returns,
        OUTPUT_DIR / "strategy_returns.csv",
    )
    save_metrics(
        xgb_sharpe,
        mlp_sharpe,
        xgb_drawdown,
        mlp_drawdown,
        xgb_ir,
        mlp_ir,
        OUTPUT_DIR / "metrics.json",
    )

    plot_cumulative_returns(
        xgb_returns,
        mlp_returns,
        benchmark,
        OUTPUT_DIR / "cumulative_returns.png",
    )
    plot_drawdown(
        xgb_returns,
        mlp_returns,
        OUTPUT_DIR / "drawdown.png",
    )
    plot_feature_importance(
        last_xgb_model,
        FEATURE_COLUMNS,
        OUTPUT_DIR / "feature_importance.png",
    )

    if last_xgb_model is not None:
        joblib.dump(last_xgb_model, OUTPUT_DIR / "xgb_model.joblib")

    print_final_comparison(
        xgb_sharpe,
        mlp_sharpe,
        xgb_drawdown,
        mlp_drawdown,
        xgb_ir,
        mlp_ir,
    )

    log(f"\nDone. Artifacts written to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nInterrupted. Partial cache may be saved under data/cache/.")
        sys.exit(130)
