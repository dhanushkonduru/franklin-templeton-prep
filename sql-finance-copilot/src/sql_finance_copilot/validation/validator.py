import sqlglot
from sqlglot.expressions import Select

FORBIDDEN = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE"
}

def validate_sql(query: str) -> bool:

    upper_query = query.upper()

    for keyword in FORBIDDEN:
        if keyword in upper_query:
            return False

    try:
        parsed = sqlglot.parse_one(query)

        if not isinstance(parsed, Select):
            return False

    except Exception:
        return False

    return True