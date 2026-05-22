from retrieval.retrieval_pipeline import (
    retrieve_context
)

def retrieval_agent(state):

    context = retrieve_context(
        state.ticker,
        "AI growth outlook and risks"
    )

    state.retrieved_context = context

    state.audit_log.append(
        "Retrieval Agent completed"
    )

    return state