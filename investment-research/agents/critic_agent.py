from utils.llm import ask_llm

def critic_agent(state):

    prompt = f"""
You are a senior investment research reviewer.

Review the following report.

Evaluate:
1. clarity
2. completeness
3. reasoning quality
4. risk coverage
5. hallucination risk

Return:
- quality score out of 10
- short critique
- APPROVED or REVISE
"""

    result = ask_llm(
        prompt + state.final_report
    )

    state.audit_log.append(
        "Critic Agent completed"
    )

    state.critic_review = result
    state.revision_count += 1

    return state