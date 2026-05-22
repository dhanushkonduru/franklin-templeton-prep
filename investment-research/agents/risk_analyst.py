from utils.llm import ask_llm
from persistence.audit_store import save_audit

def risk_analyst(state):

    prompt = f"""
You are a financial risk analyst.

Analyze possible risks for {state.company_name}.
Relevant Filing Context:
{state.retrieved_context}

Consider:
- macroeconomic risks
- AI competition
- regulation
- market risks
- operational risks

Return:
1. top risks
2. severity
3. short explanation
"""

    result = ask_llm(prompt)

    state.risk_analysis = result

    state.audit_log.append(
        "Risk Analyst completed"
    )
    
    save_audit(
    "risk_analyst",
    "Completed risk analysis"
)

    return state