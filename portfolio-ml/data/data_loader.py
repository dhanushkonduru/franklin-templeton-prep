from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def load_price_data(tickers, start, end, use_cache=True):

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_key = "_".join(sorted(tickers))
    cache_file = CACHE_DIR / f"prices_{cache_key}_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached prices from {cache_file.name}", flush=True)
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return data

    print(
        f"Downloading price data for {len(tickers)} tickers ({start} to {end})...",
        flush=True,
    )

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )["Close"]

    data.to_csv(cache_file)
    print(f"Price data downloaded ({len(data)} rows).", flush=True)

    return data
