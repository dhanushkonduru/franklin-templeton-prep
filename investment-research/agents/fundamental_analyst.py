from persistence.audit_store import save_audit

def fundamental_analyst(state):

    pe = state.fundamentals.get("pe_ratio")

    if pe and pe < 30:
        analysis = "Company appears reasonably valued."
    else:
        analysis = "Company may be highly valued."

    state.fundamentals["analysis"] = analysis

    state.audit_log.append("Fundamental Analyst completed")

    save_audit(
        "fundamental_analyst",
        "Completed fundamental analysis"
    )

    return state