from utils.llm import ask_llm
from persistence.audit_store import save_audit
from retrieval.retrieval_pipeline import (
    retrieve_context
)

def sentiment_analyst(state):

    news_text = "\n".join(state.news)

    retrieve_context(state.ticker, "AI growth outlook")

    prompt = f"""
You are a financial sentiment analyst.

Analyze the following news headlines for {state.ticker}.

Transcript Context:
{state.retrieved_context}

News:
{news_text}

Return:
1. overall sentiment
2. management confidence
3. investor outlook
4. short reasoning
"""

    result = ask_llm(prompt)

    state.sentiment_analysis = result

    state.audit_log.append(
        "Sentiment Analyst completed"
    )

    save_audit(
        "sentiment_analyst",
        "Completed sentiment analysis"
    )

    return state