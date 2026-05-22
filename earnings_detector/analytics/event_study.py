"""
EVENT STUDY ANALYTICS
---------------------
The event study is the core of "does this signal matter?"

METHODOLOGY (from finance research, specifically Fama et al. 1969):

1. Pick an event window: we use (-1, +1) — the day before earnings through
   the day after. Why? Earnings are released after market close, so:
   - Day -1: any pre-announcement leakage
   - Day  0: market reaction (after-hours + open next day)  
   - Day +1: analyst digest + follow-on moves

2. Define "abnormal return": what return can't be explained by market moves?
   AR(t) = R_stock(t) - [alpha + beta * R_market(t)]
   
   The market model: we regress stock returns on market returns in an
   ESTIMATION WINDOW (typically -252 to -11 days before the event).
   This gives us expected alpha and beta.

3. Cumulative Abnormal Return (CAR) = sum of ARs over the event window
   CAR(-1, +1) is the key metric.

4. SURPRISE DETECTION: if sentiment_score was +0.3 (positive) but CAR = -0.05
   (stock fell), that's a "sentiment surprise" — management was upbeat but
   the market disagreed. These divergences are the most interesting cases.

CAVEAT: With 200 transcripts across 20 companies, you may find correlation
but NOT causation. Always say "correlates with" not "causes". The FT interviewer
will probe this — acknowledge it first.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from typing import Dict, Optional, Tuple
import json


def fetch_prices(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV data using yfinance.
    Returns DataFrame with columns: date, open, high, low, close, volume, returns
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data for {ticker}")
            return None
        
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Daily log returns: ln(P_t / P_{t-1})
        # Why log returns? They're additive over time, approximately normal.
        # For a 3-day window, log return ≈ simple return for small values.
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['date'] = df.index
        
        return df.dropna()
        
    except ImportError:
        print("yfinance not available — returning synthetic data for demo")
        return _synthetic_price_data(ticker, start_date, end_date)
    except Exception as e:
        print(f"Price fetch error for {ticker}: {e}")
        return None


def _synthetic_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Generate synthetic prices for demo when yfinance fails."""
    np.random.seed(hash(ticker) % 2**31)
    dates = pd.date_range(start_date, end_date, freq='B')  # business days
    n = len(dates)
    
    returns = np.random.normal(0.0005, 0.015, n)  # ~0.05% daily drift, 1.5% vol
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(5_000_000, 20_000_000, n),
        'returns': returns,
    })
    df.set_index('date', inplace=True)
    return df


def estimate_market_model(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    estimation_window: Tuple[int, int] = (-252, -11)
) -> Tuple[float, float]:
    """
    Estimate OLS market model: R_i = alpha + beta * R_m + epsilon
    
    ESTIMATION WINDOW: use historical data far from the event to avoid
    contaminating the model with the very returns you're trying to explain.
    Convention: 252 trading days back (1 year) to 11 days before event.
    
    Returns: (alpha, beta)
    """
    from scipy import stats
    
    # Align series
    aligned = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()
    
    if len(aligned) < 30:
        print("Warning: insufficient data for market model, using beta=1, alpha=0")
        return 0.0, 1.0
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        aligned['market'], aligned['stock']
    )
    
    return intercept, slope  # alpha, beta


def compute_car(
    ticker: str,
    event_date: datetime,
    sentiment_score: float,
    market_ticker: str = "SPY",
    event_window: Tuple[int, int] = (-1, 1),
    estimation_window: Tuple[int, int] = (-252, -11)
) -> Dict:
    """
    Full event study for one earnings call.
    
    Returns dict with:
      - car: cumulative abnormal return over event window
      - abnormal_returns: day-by-day ARs
      - alpha, beta: market model parameters  
      - sentiment_surprise: divergence between sentiment and price reaction
    """
    # Date range for data fetch
    fetch_start = (event_date + timedelta(days=estimation_window[0] - 5)).strftime('%Y-%m-%d')
    fetch_end   = (event_date + timedelta(days=event_window[1] + 5)).strftime('%Y-%m-%d')
    
    stock_df  = fetch_prices(ticker, fetch_start, fetch_end)
    market_df = fetch_prices(market_ticker, fetch_start, fetch_end)
    
    if stock_df is None or market_df is None:
        return {'error': 'price data unavailable'}
    
    # Align on dates
    merged = stock_df[['returns']].join(
        market_df[['returns']].rename(columns={'returns': 'market_returns'}),
        how='inner'
    )
    
    # Find event date index
    try:
        event_idx = merged.index.get_indexer([event_date], method='nearest')[0]
    except Exception:
        return {'error': 'event date not in price data'}
    
    # Estimation window
    est_start = max(0, event_idx + estimation_window[0])
    est_end   = max(0, event_idx + estimation_window[1])
    est_data  = merged.iloc[est_start:est_end]
    
    alpha, beta = estimate_market_model(est_data['returns'], est_data['market_returns'])
    
    # Event window abnormal returns
    evt_start = max(0, event_idx + event_window[0])
    evt_end   = min(len(merged) - 1, event_idx + event_window[1] + 1)
    evt_data  = merged.iloc[evt_start:evt_end]
    
    expected_returns = alpha + beta * evt_data['market_returns']
    abnormal_returns = evt_data['returns'] - expected_returns
    car = abnormal_returns.sum()
    
    # Sentiment surprise: if sentiment was positive but CAR is negative,
    # or sentiment was negative but CAR is positive → SURPRISE
    # We measure this as the divergence: positive sentiment_score, negative CAR = negative surprise
    sentiment_surprise = sentiment_score * np.sign(car)  # +1 = aligned, -1 = divergent
    
    return {
        'ticker': ticker,
        'event_date': event_date.strftime('%Y-%m-%d'),
        'alpha': round(alpha, 6),
        'beta': round(beta, 4),
        'car': round(float(car), 6),
        'car_pct': round(float(car) * 100, 3),
        'abnormal_returns': {
            str(date.date()): round(float(ar), 6)
            for date, ar in abnormal_returns.items()
        },
        'sentiment_score': round(sentiment_score, 4),
        'sentiment_surprise': round(sentiment_surprise, 4),
        'is_divergent': abs(sentiment_score) > 0.1 and np.sign(car) != np.sign(sentiment_score),
        'n_estimation_days': len(est_data),
    }


def detect_surprise(sentiment_score: float, eps_actual: float, eps_estimate: float,
                    car: float) -> Dict:
    """
    SURPRISE DETECTION: Three-way signal combining:
    1. Earnings surprise: (actual - estimate) / |estimate|
    2. Sentiment score from NLP
    3. Price reaction (CAR)
    
    The most interesting case is when TWO diverge:
    - Management positive + EPS beat + stock falls → priced in, or credibility gap
    - Management negative + EPS miss + stock rises → bar was low, kitchen-sink quarter
    - Management positive + EPS beat + stock rises → clean beat (boring)
    """
    eps_surprise = (eps_actual - eps_estimate) / abs(eps_estimate) if eps_estimate != 0 else 0
    
    # Classify each dimension
    sentiment_dir = "positive" if sentiment_score > 0.05 else ("negative" if sentiment_score < -0.05 else "neutral")
    eps_dir = "beat" if eps_surprise > 0.02 else ("miss" if eps_surprise < -0.02 else "in-line")
    price_dir = "up" if car > 0.01 else ("down" if car < -0.01 else "flat")
    
    # Divergence patterns
    divergent = (
        (sentiment_dir == "positive" and price_dir == "down") or
        (sentiment_dir == "negative" and price_dir == "up") or
        (eps_dir == "beat" and price_dir == "down") or
        (eps_dir == "miss" and price_dir == "up")
    )
    
    return {
        'eps_surprise_pct': round(eps_surprise * 100, 2),
        'sentiment_direction': sentiment_dir,
        'eps_direction': eps_dir,
        'price_direction': price_dir,
        'is_divergent': divergent,
        'divergence_type': _classify_divergence(sentiment_dir, eps_dir, price_dir) if divergent else None,
    }


def _classify_divergence(sentiment: str, eps: str, price: str) -> str:
    if sentiment == "positive" and price == "down":
        return "CREDIBILITY_GAP"     # Management was upbeat, market didn't buy it
    if sentiment == "negative" and price == "up":
        return "KITCHEN_SINK"        # Management was cautious, bar was already low
    if eps == "beat" and price == "down":
        return "PRICED_IN"           # Beat was expected, or guidance disappointed
    if eps == "miss" and price == "up":
        return "LOW_BAR"             # Miss was expected / guidance was better
    return "COMPLEX_DIVERGENCE"


def run_event_study_demo():
    """Demo run showing the full pipeline output."""
    print("=" * 60)
    print("EVENT STUDY DEMO — Apple Q4 2024")
    print("=" * 60)
    
    # Synthetic sentiment score (what FinBERT would output)
    sentiment_score = 0.28  # net positive
    
    result = compute_car(
        ticker="AAPL",
        event_date=datetime(2024, 11, 1),
        sentiment_score=sentiment_score,
        event_window=(-1, 1),
    )
    
    print(f"\nMarket model: alpha={result.get('alpha')}, beta={result.get('beta')}")
    print(f"CAR(-1,+1):   {result.get('car_pct')}%")
    print(f"Sentiment:    {sentiment_score:+.2f} ({result.get('car_pct',0):+.2f}% price reaction)")
    print(f"Divergent?    {result.get('is_divergent')}")
    
    if result.get('abnormal_returns'):
        print("\nAbnormal returns by day:")
        for date, ar in result['abnormal_returns'].items():
            bar = "█" * int(abs(ar) * 200) if abs(ar) > 0.001 else ""
            sign = "+" if ar >= 0 else ""
            print(f"  {date}: {sign}{ar*100:.2f}% {bar}")
    
    # Surprise detection
    surprise = detect_surprise(
        sentiment_score=sentiment_score,
        eps_actual=1.64,
        eps_estimate=1.60,
        car=result.get('car', 0)
    )
    print("\nSurprise analysis:")
    for k, v in surprise.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_event_study_demo()
