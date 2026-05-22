"""Generate realistic synthetic financial statement data and insert into Postgres.

This script:
- Reflects the `stocks` and `financials` tables (expects them to exist).
- Creates `financial_metrics` table if missing and upserts EPS, PE ratio and debt-to-equity.
- Generates multiple years of annual statements per ticker with realistic growth/margins.
- Uses SQLAlchemy Core and batch upserts.

Usage:
  python scripts/generate_synthetic_financials.py --db-url postgresql://user:pass@localhost:5432/db --years 5
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from collections import defaultdict
from decimal import Decimal
from datetime import date
from typing import Dict, List

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    BigInteger,
    Numeric,
    Date,
    String,
    TIMESTAMP,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert


LOG = logging.getLogger("generate_synthetic_financials")


def setup_logging(level=logging.INFO):
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(h)


def make_metrics_table(metadata: MetaData) -> Table:
    return Table(
        "financial_metrics",
        metadata,
        Column("metric_id", BigInteger, primary_key=True),
        Column("stock_id", Integer, nullable=False),
        Column("period_end", Date, nullable=False),
        Column("eps", Numeric(20,6)),
        Column("pe_ratio", Numeric(10,4)),
        Column("debt_to_equity", Numeric(10,6)),
        Column("created_at", TIMESTAMP),
    )


def deterministic_seed(ticker: str) -> random.Random:
    h = abs(hash(ticker)) % (2 ** 32)
    return random.Random(h)


def generate_company_baseline(rng: random.Random):
    # Choose size bucket and base revenue (USD)
    bucket = rng.choices(["small", "mid", "large"], weights=[0.3, 0.5, 0.2])[0]
    if bucket == "small":
        base_revenue = rng.uniform(50e6, 800e6)
    elif bucket == "mid":
        base_revenue = rng.uniform(0.8e9, 8e9)
    else:
        base_revenue = rng.uniform(8e9, 200e9)

    shares_outstanding = int(rng.uniform(50e6, 5e9))
    return base_revenue, shares_outstanding


def generate_annual_series(ticker: str, years: int) -> List[Dict]:
    rng = deterministic_seed(ticker)
    base_revenue, shares_outstanding = generate_company_baseline(rng)
    series = []
    current_year = date.today().year
    revenue = base_revenue
    for i in range(years, 0, -1):
        fiscal_year = current_year - i + 1
        # growth between -10% and +30% (bounded)
        growth = max(-0.15, min(0.35, rng.normalvariate(0.08, 0.12)))
        revenue = revenue * (1 + growth)

        # net margin realistic: -0.2..0.3 but center around 0.08
        net_margin = max(-0.2, min(0.35, rng.normalvariate(0.09, 0.08)))
        net_income = revenue * net_margin

        # assets to revenue ratio and liabilities ratio
        assets_to_rev = max(0.5, min(8.0, rng.normalvariate(2.0, 1.0)))
        total_assets = revenue * assets_to_rev
        liabilities_ratio = max(0.05, min(0.9, rng.normalvariate(0.45, 0.15)))
        total_liabilities = total_assets * liabilities_ratio

        # EPS
        eps = None
        if shares_outstanding > 0:
            eps = Decimal(net_income / shares_outstanding) if shares_outstanding else None

        # PE ratio: if EPS positive choose 5..40, else None
        pe = None
        if eps is not None and eps > 0:
            pe = Decimal(rng.uniform(5.0, 40.0))

        # debt-to-equity
        equity = total_assets - total_liabilities
        debt_to_equity = None
        if equity and equity > 0:
            debt_to_equity = Decimal(total_liabilities / equity)

        period_end = date(fiscal_year, 12, 31)

        series.append(
            {
                "fiscal_year": fiscal_year,
                "period_end": period_end,
                "revenue": Decimal(round(revenue, 2)),
                "net_income": Decimal(round(net_income, 2)),
                "shares_outstanding": shares_outstanding,
                "eps": (eps.quantize(Decimal("0.000001")) if eps is not None else None),
                "pe_ratio": (pe.quantize(Decimal("0.0001")) if pe is not None else None),
                "total_assets": Decimal(round(total_assets, 2)),
                "total_liabilities": Decimal(round(total_liabilities, 2)),
                "debt_to_equity": (debt_to_equity.quantize(Decimal("0.000001")) if debt_to_equity is not None else None),
            }
        )

    return series


def upsert_financials_and_metrics(engine, financials_table: Table, metrics_table: Table, stock_id: int, series: List[Dict]):
    fin_rows = []
    metrics_rows = []
    for row in series:
        fin_rows.append(
            {
                "stock_id": stock_id,
                "period_end": row["period_end"],
                "fiscal_year": row["fiscal_year"],
                "fiscal_quarter": None,
                "statement_type": "income",
                "revenue": row["revenue"],
                "gross_profit": None,
                "operating_income": None,
                "net_income": row["net_income"],
                "total_assets": row["total_assets"],
                "total_liabilities": row["total_liabilities"],
                "cash_and_equivalents": None,
                "shares_outstanding": row["shares_outstanding"],
            }
        )

        metrics_rows.append(
            {
                "stock_id": stock_id,
                "period_end": row["period_end"],
                "eps": row["eps"],
                "pe_ratio": row["pe_ratio"],
                "debt_to_equity": row["debt_to_equity"],
            }
        )

    # Upsert financials (unique constraint on stock_id, period_end, statement_type)
    fin_stmt = pg_insert(financials_table).values(fin_rows)
    fin_upsert = fin_stmt.on_conflict_do_update(
        index_elements=[financials_table.c.stock_id, financials_table.c.period_end, financials_table.c.statement_type],
        set_={
            "revenue": fin_stmt.excluded.revenue,
            "net_income": fin_stmt.excluded.net_income,
            "total_assets": fin_stmt.excluded.total_assets,
            "total_liabilities": fin_stmt.excluded.total_liabilities,
            "shares_outstanding": fin_stmt.excluded.shares_outstanding,
            "fiscal_year": fin_stmt.excluded.fiscal_year,
        },
    )

    # Upsert metrics (unique on stock_id, period_end) — create index if needed
    met_stmt = pg_insert(metrics_table).values(metrics_rows)
    met_upsert = met_stmt.on_conflict_do_update(
        index_elements=[metrics_table.c.stock_id, metrics_table.c.period_end],
        set_={
            "eps": met_stmt.excluded.eps,
            "pe_ratio": met_stmt.excluded.pe_ratio,
            "debt_to_equity": met_stmt.excluded.debt_to_equity,
        },
    )

    with engine.begin() as conn:
        LOG.info("Upserting %d financial rows for stock_id=%s", len(fin_rows), stock_id)
        conn.execute(fin_upsert)
        LOG.info("Upserting %d metric rows for stock_id=%s", len(metrics_rows), stock_id)
        conn.execute(met_upsert)


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic financial statement data")
    p.add_argument("--db-url", default=os.environ.get("DATABASE_URL"), help="SQLAlchemy DB URL")
    p.add_argument("--tickers", default=None, help="Comma-separated tickers to generate (defaults to all stocks table)")
    p.add_argument("--years", type=int, default=5, help="Number of years to generate per ticker")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    if not args.db_url:
        LOG.error("No DB URL provided. Set --db-url or DATABASE_URL env var.")
        raise SystemExit(2)

    engine = create_engine(args.db_url)
    metadata = MetaData()
    # reflect existing tables
    metadata.reflect(bind=engine, only=["stocks", "financials"]) 
    stocks_table = metadata.tables.get("stocks")
    financials_table = metadata.tables.get("financials")
    if not stocks_table or not financials_table:
        LOG.error("Expected `stocks` and `financials` tables to exist")
        raise SystemExit(2)

    # create metrics table if missing
    metrics_table = make_metrics_table(metadata)
    metadata.create_all(bind=engine, tables=[metrics_table])

    # decide tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        # map to stock ids
        with engine.connect() as conn:
            q = select([stocks_table.c.stock_id, stocks_table.c.ticker]).where(
                stocks_table.c.ticker.in_(tickers)
            )
            rows = conn.execute(q).fetchall()
        mapping = {r.ticker: r.stock_id for r in rows}
        missing = [t for t in tickers if t not in mapping]
        if missing:
            LOG.warning("Tickers not found in `stocks` table and will be skipped: %s", missing)
        tickers = [t for t in tickers if t in mapping]
    else:
        # use all tickers in stocks table
        with engine.connect() as conn:
            q = select([stocks_table.c.stock_id, stocks_table.c.ticker])
            rows = conn.execute(q).fetchall()
        mapping = {r.ticker: r.stock_id for r in rows}
        tickers = list(mapping.keys())

    LOG.info("Generating synthetic financials for %d tickers (%d years each)", len(tickers), args.years)

    # generate and insert per ticker
    for t in tickers:
        stock_id = mapping[t]
        series = generate_annual_series(t, args.years)
        upsert_financials_and_metrics(engine, financials_table, metrics_table, stock_id, series)

    LOG.info("Done generating synthetic financials")


if __name__ == "__main__":
    main()
