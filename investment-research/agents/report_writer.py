from utils.llm import ask_llm
from persistence.audit_store import save_audit

def report_writer(state):

    prompt = f"""
You are a senior investment analyst.

Generate a professional investment report.

Use retrieved evidence and cite important findings.

Ticker:
{state.ticker}

Fundamentals:
{state.fundamentals}

Sentiment:
{state.sentiment_analysis}

Risks:
{state.risk_analysis}

Return:
- executive summary
- bullish factors
- bearish factors
- recommendation
"""

    report = ask_llm(prompt)

    state.final_report = report

    state.audit_log.append(
        "Report Writer completed"
    )
    
    save_audit(
    "report_writer",
    "Generated final report"
)

    return state