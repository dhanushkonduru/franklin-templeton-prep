import yfinance as yf

def fetch_fundamentals(tickers):

    fundamentals = {}

    for ticker in tickers:

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

            print(f"Error fetching {ticker}: {e}")

    return fundamentals