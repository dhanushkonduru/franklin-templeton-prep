import yfinance as yf

def get_stock_data(ticker: str):
    stock = yf.Ticker(ticker)

    info = stock.info

    return {
        "company_name": info.get("longName"),
        "stock_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE")
    }