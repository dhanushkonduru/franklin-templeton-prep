import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

engine = create_engine("sqlite:///finance.db")

tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "META",
    "GOOGL"
]

all_data = []

for ticker in tickers:

    df = yf.download(
        ticker,
        start="2023-01-01",
        end="2024-01-01",
        auto_adjust=False
    )

    df.reset_index(inplace=True)

    # flatten column names safely
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    # lowercase
    df.columns = [
        col.lower().replace(" ", "_")
        for col in df.columns
    ]

    df["ticker"] = ticker

    all_data.append(df)

final_df = pd.concat(
    all_data,
    ignore_index=True
)

final_df.to_sql(
    "daily_prices",
    engine,
    if_exists="replace",
    index=False
)

stocks = pd.DataFrame([
    {
        "ticker": "AAPL",
        "company_name": "Apple",
        "sector": "Tech"
    },
    {
        "ticker": "MSFT",
        "company_name": "Microsoft",
        "sector": "Tech"
    },
    {
        "ticker": "NVDA",
        "company_name": "NVIDIA",
        "sector": "Tech"
    },
    {
        "ticker": "META",
        "company_name": "Meta",
        "sector": "Tech"
    },
    {
        "ticker": "GOOGL",
        "company_name": "Google",
        "sector": "Tech"
    }
])

stocks.to_sql(
    "stocks",
    engine,
    if_exists="replace",
    index=False
)

print("Database seeded successfully.")