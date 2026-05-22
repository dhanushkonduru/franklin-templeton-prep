from orchestration.graph import app

initial_state = {
    "ticker": "NVDA"
}

config = {
    "configurable": {
        "thread_id": "nvda_research_run"
    }
}

result = app.invoke(
    initial_state,
    config=config
)

print("\nFINAL REPORT\n")

print(result["final_report"])

print("\nAUDIT LOG\n")

for log in result["audit_log"]:
    print("-", log)