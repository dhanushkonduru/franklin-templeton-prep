from tools import get_stock_data

def data_agent(state):
    data = get_stock_data(state.ticker)

    state.company_name = data["company_name"]
    state.stock_price = data["stock_price"]

    return state


def sentiment_agent(state):
    if state.stock_price > 200:
        state.sentiment = "Positive"
    else:
        state.sentiment = "Neutral"

    return state


def report_agent(state):
    state.report = f"""
    Investment Report for {state.company_name}

    Stock Price: {state.stock_price}

    Market Sentiment: {state.sentiment}

    Recommendation:
    {'BUY' if state.sentiment == 'Positive' else 'HOLD'}
    """

    return state