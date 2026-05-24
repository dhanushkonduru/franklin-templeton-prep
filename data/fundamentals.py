import json
from pathlib import Path

import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def fetch_fundamentals(tickers, use_cache=True):

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_key = "_".join(sorted(tickers))
    cache_file = CACHE_DIR / f"fundamentals_{cache_key}.json"

    if use_cache and cache_file.exists():
        print(f"Loading cached fundamentals from {cache_file.name}", flush=True)
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    fundamentals = {}
    total = len(tickers)

    print(f"Fetching fundamentals for {total} tickers...", flush=True)

    for i, ticker in enumerate(tickers, start=1):

        print(f"  [{i}/{total}] {ticker}", flush=True)

        try:

            stock = yf.Ticker(ticker)

            info = stock.info

            fundamentals[ticker] = {

                "pe_ratio":
                    info.get("trailingPE"),

                "roe":
                    info.get("returnOnEquity"),

                "debt_to_equity":
                    info.get("debtToEquity"),

                "revenue_growth":
                    info.get("revenueGrowth")
            }

        except Exception as e:

            print(f"  Error fetching {ticker}: {e}", flush=True)

            fundamentals[ticker] = {}

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(fundamentals, f)

    print("Fundamentals fetch complete.", flush=True)

    return fundamentals
