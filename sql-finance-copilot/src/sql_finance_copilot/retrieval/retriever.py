from typing import List

SCHEMA_MAP = {
    "stocks": """
    Table: stocks
    Columns:
    - ticker
    - company_name
    - sector
    - industry
    - market_cap

    Known values:
    - sector: Tech
    """,

    "daily_prices": """
    Table: daily_prices
    Columns:
    - ticker
    - date
    - open
    - high
    - low
    - close
    - volume
    """,

    "financials": """
    Table: financials
    Columns:
    - ticker
    - year
    - revenue
    - net_income
    - eps
    - pe_ratio
    """
}

def retrieve_relevant_schema(question: str) -> str:

    question_lower = question.lower()

    relevant_tables: List[str] = []

    if any(word in question_lower for word in [
    "price",
    "stock",
    "volume",
    "return",
    "performance"
    ]):
        relevant_tables.append("daily_prices")

    if any(word in question_lower for word in [
        "sector",
        "company",
        "tech"
    ]):
        relevant_tables.append("stocks")

    if any(word in question_lower for word in [
        "revenue",
        "eps",
        "pe_ratio",
        "financial"
    ]):
        relevant_tables.append("financials")

    if not relevant_tables:
        relevant_tables = ["stocks"]

    schema_text = "\n\n".join(
        SCHEMA_MAP[table]
        for table in relevant_tables
    )

    return schema_text