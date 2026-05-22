# Portfolio Construction with Classical ML

This project answers the research question:

**Can classical machine learning be used to predict cross-sectional stock returns and build a profitable long-short portfolio, while avoiding look-ahead bias?**

The repository implements a compact financial ML pipeline that uses historical price data and fundamental data to predict forward returns, rank stocks each period, and evaluate a long-short strategy with walk-forward validation.

## What the project does

The codebase follows a full research loop:

1. Download historical prices for a small basket of liquid large-cap stocks.
2. Fetch fundamental ratios such as P/E, ROE, debt-to-equity, and revenue growth.
3. Engineer time-series features such as lagged returns, 3-month momentum, 12-month momentum, and rolling volatility.
4. Build a supervised dataset where the target is the next 21 trading-day return.
5. Train two models:
	- an XGBoost regressor inside an sklearn `Pipeline`
	- a small PyTorch MLP baseline
6. Evaluate the models with expanding-window walk-forward validation.
7. Convert predictions into long-short portfolios by going long the top decile and short the bottom decile.
8. Measure strategy quality using Sharpe ratio, max drawdown, and information ratio.
9. Apply a simple transaction cost adjustment.
10. Add a regime label with KMeans as a lightweight regime-detection layer.

## Why this question matters

This is a classic financial ML problem, and it is a good interview topic because it tests more than model selection. The hard part is not fitting a model; it is building a research process that survives the realities of financial data:

- the signal-to-noise ratio is low
- markets are non-stationary
- random cross-validation leaks future information
- transaction costs can destroy paper alpha
- feature redundancy is common in factor-style data

## Approach

### Data

The current implementation uses Yahoo Finance data through `yfinance`.

- Price data is loaded for a small list of large-cap tickers.
- Fundamental data is fetched per ticker from Yahoo Finance metadata.
- The target is the forward 21 trading-day return, which approximates a one-month horizon.

### Features

The feature set is intentionally simple and interpretable:

- `return_1d`
- `momentum_3m`
- `momentum_12m`
- `volatility`
- `pe_ratio`
- `roe`
- `debt_to_equity`
- `revenue_growth`

This mixes market-based and fundamental signals so the model can learn cross-sectional relationships similar to value, momentum, quality, and risk factors.

### Models

The repo compares two model families:

- **XGBoost regressor**: the main classical ML model, wrapped in an sklearn `Pipeline` with a `ColumnTransformer` and median imputation plus standard scaling.
- **MLP regressor**: a small PyTorch neural network used as a simple deep learning baseline.

The goal is not to claim that one model always wins. The point is to compare a strong tabular baseline against a lightweight neural network on the same walk-forward splits.

### Validation

The validation scheme is expanding-window walk-forward testing.

That means the model trains on years 1 to N, tests on year N+1, then expands the training set and repeats. This is the correct choice for financial ML because random k-fold validation can leak information from the future into the past.

### Portfolio construction

Predictions are converted into a cross-sectional ranking each period.

- top 10% of predictions = long
- bottom 10% of predictions = short
- the rest = flat

This creates a simple long-short portfolio that focuses on ranking quality rather than absolute return prediction.

### Performance metrics

The strategy is evaluated with:

- **Sharpe ratio** to measure risk-adjusted return
- **Max drawdown** to measure downside severity
- **Information ratio** to compare the strategy with a benchmark

## How it works end to end

The main entry point is `main.py`.

1. `data/data_loader.py` downloads adjusted close prices.
2. `data/fundamentals.py` fetches company fundamentals.
3. `features/feature_engineering.py` creates return, momentum, and volatility features.
4. `features/targets.py` creates the forward return target.
5. `features/build_dataset.py` merges price features, fundamentals, and targets into one table.
6. `models/regime_detection.py` assigns a simple market regime label with KMeans.
7. `evaluation/backtest_engine.py` runs expanding-window walk-forward training and prediction.
8. `models/train_pipeline.py` builds and fits the XGBoost pipeline.
9. `models/train_mlp.py` trains the PyTorch MLP.
10. `models/predict_mlp.py` generates MLP predictions.
11. `backtest/portfolio.py` maps predictions into long and short positions.
12. `evaluation/performance.py` computes strategy returns and applies transaction costs.
13. `evaluation/metrics.py` calculates Sharpe ratio, drawdown, and information ratio.

## Why walk-forward instead of random k-fold

Random k-fold is usually wrong for market data.

In finance, observations are time ordered. If future data appears in training folds, the model gets an unrealistic preview of market conditions. That creates look-ahead bias, which is one of the most common reasons backtests look better than live results.

Walk-forward validation avoids this by only allowing past data to predict the future.

## Bias, variance, and regularization

Financial ML usually has a brutal bias-variance tradeoff because the dataset is noisy and features are correlated.

- **High variance** models can overfit short-lived patterns that disappear quickly.
- **High bias** models can miss weak but real signals.
- **L1 regularization** is useful when many features are redundant.
- **L2 regularization** helps stabilize weights when signals are correlated.

In practice, the best strategy is usually to keep the model simple, validate it properly, and test whether the signal survives transaction costs.

## What was kept in the cleaned repo

The cleanup keeps the core implementation that is actually used by the pipeline:

- data ingestion
- feature engineering
- target generation
- dataset building
- XGBoost training
- MLP training and prediction
- regime detection
- backtesting
- portfolio construction
- performance metrics

Unused helper code that was not wired into the main workflow was removed, and the dependency list was trimmed to match the actual runtime requirements.

## Installation

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

## Notes and limitations

- The current code uses a small demonstration universe rather than the full S&P 500.
- The target is a 21 trading-day forward return, so the strategy is effectively monthly horizon.
- The regime detection step is intentionally lightweight and can be replaced with a more advanced HMM or macro-driven regime model.
- SHAP explainability was removed during cleanup because it was not connected to the main research loop.

## Interview-ready summary

You can describe the project like this:

"I built a classical financial ML pipeline for cross-sectional return prediction. I used price and fundamental data, engineered momentum and volatility features, trained an XGBoost model and a small MLP, and evaluated them with expanding-window walk-forward validation. I then converted predictions into a long-short portfolio and measured Sharpe ratio, drawdown, and information ratio. The main lesson was that the modeling problem is secondary to preventing look-ahead bias and controlling for regime change and transaction costs."