"""Seed PostgreSQL financial tables using yfinance.

Usage:
  python scripts/seed_postgres.py --db-url postgresql://user:pass@localhost:5432/db

Features:
- Uses SQLAlchemy Core to reflect `stocks` and `daily_prices` tables.
- Batch upserts for `stocks` (by `ticker`) and batch inserts for `daily_prices`.
- Avoids duplicate daily rows via `ON CONFLICT DO NOTHING` on `(stock_id, trade_date)`.
- Retries yfinance fetches with exponential backoff and logs failures.
- Chunked inserts for large volumes.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf
from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

LOG = logging.getLogger("seed_postgres")


DEFAULT_TICKERS = [
    # tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    # financials
    "JPM",
    "BAC",
    "GS",
    "MS",
    # energy
    "XOM",
    "CVX",
    # healthcare
    "JNJ",
    "PFE",
    "UNH",
    # consumer
    "KO",
    "PEP",
    "WMT",
    "DIS",
    # others
    "V",
    "MA",
    "NFLX",
]

SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Technology",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "UNH": "Healthcare",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "WMT": "Consumer Staples",
    "DIS": "Communication Services",
    "V": "Financials",
    "MA": "Financials",
    "NFLX": "Communication Services",
}


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def retry(max_attempts: int = 3, base_sleep: float = 1.0):
    def deco(fn):
        def wrapper(*a, **kw):
            attempt = 0
            while True:
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        LOG.exception("Giving up after %d attempts", attempt)
                        raise
                    sleep = base_sleep * (2 ** (attempt - 1))
                    LOG.warning("Attempt %d failed: %s — retrying in %.1fs", attempt, e, sleep)
                    time.sleep(sleep)

        return wrapper

    return deco


@retry(max_attempts=3, base_sleep=1.0)
def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    LOG.info("Fetching %s from %s to %s", ticker, start, end)
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return df


def upsert_stocks(engine, stocks_table, stock_rows: List[Dict]) -> None:
    stmt = pg_insert(stocks_table).values(stock_rows)
    # On conflict on ticker, update mutable fields
    do_update = stmt.on_conflict_do_update(
        index_elements=[stocks_table.c.ticker],
        set_={
            "name": stmt.excluded.name,
            "exchange": stmt.excluded.exchange,
            "sector": stmt.excluded.sector,
            "industry": stmt.excluded.industry,
            "currency": stmt.excluded.currency,
            "ipo_date": stmt.excluded.ipo_date,
        },
    )
    with engine.begin() as conn:
        LOG.info("Upserting %d stocks", len(stock_rows))
        conn.execute(do_update)


def batch_insert_daily_prices(engine, daily_table, rows: List[Dict], chunk_size: int = 1000) -> int:
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            stmt = pg_insert(daily_table).values(chunk).on_conflict_do_nothing(
                index_elements=[daily_table.c.stock_id, daily_table.c.trade_date]
            )
            res = conn.execute(stmt)
            # res.rowcount can be -1 depending on driver; we approximate with len(chunk)
            inserted += len(chunk)
            LOG.info("Inserted chunk %d (%d rows)", i // chunk_size + 1, len(chunk))
    return inserted


def seed(db_url: str, tickers: List[str], start: str, end: str) -> None:
    engine = create_engine(db_url)
    meta = MetaData()
    meta.reflect(bind=engine, only=["stocks", "daily_prices"])  # keep limited

    stocks_table = meta.tables.get("stocks")
    daily_table = meta.tables.get("daily_prices")
    if stocks_table is None or daily_table is None:
        raise RuntimeError("Expected `stocks` and `daily_prices` tables to exist in DB")

    # Prepare stock rows for upsert
    stock_rows = []
    for t in tickers:
        row = {
            "ticker": t,
            "name": None,
            "exchange": None,
            "sector": SECTOR_MAP.get(t, None),
            "industry": None,
            "currency": "USD",
        }
        stock_rows.append(row)

    upsert_stocks(engine, stocks_table, stock_rows)

    # Map tickers to stock_id
    with engine.connect() as conn:
        q = select([stocks_table.c.stock_id, stocks_table.c.ticker]).where(
            stocks_table.c.ticker.in_(tickers)
        )
        rows = conn.execute(q).fetchall()
    ticker_to_id = {r.ticker: r.stock_id for r in rows}

    all_price_rows = []
    for t in tickers:
        try:
            df = fetch_history(t, start, end)
        except Exception:
            LOG.exception("Failed to fetch %s — skipping", t)
            continue

        # Normalize DataFrame columns
        df = df.rename(columns={
            c: c.lower().replace(" ", "_") for c in df.columns
        })
        # required columns: open, high, low, close, adj_close, volume
        for idx, row in df.iterrows():
            # index -> Timestamp
            trade_date = pd.Timestamp(idx).date()
            try:
                price_row = {
                    "stock_id": int(ticker_to_id[t]),
                    "trade_date": trade_date,
                    "open": float(row.get("open")) if pd.notna(row.get("open")) else None,
                    "high": float(row.get("high")) if pd.notna(row.get("high")) else None,
                    "low": float(row.get("low")) if pd.notna(row.get("low")) else None,
                    "close": float(row.get("close")) if pd.notna(row.get("close")) else None,
                    "adj_close": float(row.get("adj_close")) if pd.notna(row.get("adj_close")) else None,
                    "volume": int(row.get("volume")) if pd.notna(row.get("volume")) else None,
                }
            except Exception:
                LOG.exception("Skipping malformed row for %s on %s", t, trade_date)
                continue
            all_price_rows.append(price_row)

    if not all_price_rows:
        LOG.warning("No price rows to insert")
        return

    LOG.info("Total daily rows to insert: %d", len(all_price_rows))
    inserted = batch_insert_daily_prices(engine, daily_table, all_price_rows)
    LOG.info("Finished inserting daily prices (approx %d rows processed)", inserted)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed PostgreSQL with financial data via yfinance")
    p.add_argument("--db-url", default=os.environ.get("DATABASE_URL"), help="SQLAlchemy DB URL")
    p.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (defaults to today)")
    p.add_argument("--tickers", default=None, help="Comma-separated tickers to seed (overrides default set)")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if not args.db_url:
        LOG.error("No DB URL provided. Set --db-url or DATABASE_URL env var.")
        raise SystemExit(2)

    tickers = DEFAULT_TICKERS
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    end = args.end
    if end is None:
        end = pd.Timestamp("today").date().isoformat()

    try:
        seed(args.db_url, tickers, args.start, end)
    except Exception:
        LOG.exception("Seeding failed")
        raise


if __name__ == "__main__":
    main()
