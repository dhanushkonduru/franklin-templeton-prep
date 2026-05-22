from graph import app

initial_state = {
    "ticker": "AAPL"
}

result = app.invoke(initial_state)

print(result["report"])