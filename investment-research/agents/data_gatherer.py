from tools.finance_tools import get_stock_data
from tools.news_tools import get_news
from persistence.audit_store import save_audit

def data_gatherer(state):

    stock_data = get_stock_data(state.ticker)

    state.company_name = stock_data["company_name"]
    state.stock_price = stock_data["stock_price"]

    state.fundamentals = stock_data

    state.news = get_news(state.ticker)

    state.audit_log.append("Data Gatherer completed")
    
    save_audit(
    "data_gatherer",
    "Fetched stock and news data"
)

    return state