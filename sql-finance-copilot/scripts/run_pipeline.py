from sql_finance_copilot.retrieval.retriever import retrieve_relevant_schema
from sql_finance_copilot.llm.sql_generator import generate_sql
from sql_finance_copilot.validation.validator import validate_sql
from sql_finance_copilot.execution.executor import execute_query
from sql_finance_copilot.repair.repair_loop import repair_query


QUESTION = input("\nEnter your financial question: ")


print("\nUSER QUESTION:")
print(QUESTION)

# STEP 1 — RETRIEVE SCHEMA
schema = retrieve_relevant_schema(QUESTION)

print("\nRETRIEVED SCHEMA:")
print(schema)

# STEP 2 — GENERATE SQL
sql = generate_sql(
    question=QUESTION,
    schema=schema
)

print("\nGENERATED SQL:")
print(sql)

# STEP 3 — VALIDATE SQL
valid = validate_sql(sql)

print("\nVALIDATION RESULT:")
print(valid)

if not valid:
    raise Exception("SQL validation failed")

# STEP 4 — EXECUTE
try:
    results = execute_query(sql)

except Exception as e:
    print("\nQUERY FAILED")
    print(str(e))

    repaired_sql = repair_query(
        sql=sql,
        error=str(e),
        schema=schema
    )

    print("\nREPAIRED SQL:")
    print(repaired_sql)

    results = execute_query(repaired_sql)

print("\nFINAL RESULTS:")
print(results)