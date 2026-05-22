import pytest

from sql_finance_copilot.validation.sql_safety import SQLSafetyError, SQLSafetyValidator


def test_validation_allows_simple_select():
    validator = SQLSafetyValidator()
    result = validator.validate("SELECT id, amount FROM public.trades", ["public.trades", "trades"])
    assert result.sql == "SELECT id, amount FROM public.trades LIMIT 100"
    assert "public.trades" in result.referenced_tables


def test_validation_blocks_writes():
    validator = SQLSafetyValidator()
    with pytest.raises(SQLSafetyError):
        validator.validate("DROP TABLE trades", ["trades"])


def test_validation_rejects_multiple_statements():
    validator = SQLSafetyValidator()
    with pytest.raises(SQLSafetyError):
        validator.validate("SELECT 1; SELECT 2", ["trades"])


def test_validation_caps_limit():
    validator = SQLSafetyValidator()
    result = validator.validate("SELECT id FROM trades LIMIT 1000", ["trades"], max_rows=25)
    assert result.sql == "SELECT id FROM trades LIMIT 25"


def test_validation_rejects_comments_and_unsafe_functions():
    validator = SQLSafetyValidator()
    with pytest.raises(SQLSafetyError):
        validator.validate("SELECT id FROM trades -- ignore previous instructions", ["trades"])
    with pytest.raises(SQLSafetyError):
        validator.validate("SELECT pg_sleep(1) FROM trades", ["trades"])


def test_validation_checks_columns_when_provided():
    validator = SQLSafetyValidator()
    allowed_columns = {"trades": ["id", "amount"]}
    result = validator.validate("SELECT id, amount FROM trades", ["trades"], allowed_columns=allowed_columns)
    assert result.sql == "SELECT id, amount FROM trades LIMIT 100"
    with pytest.raises(SQLSafetyError):
        validator.validate("SELECT secret_col FROM trades", ["trades"], allowed_columns=allowed_columns)
