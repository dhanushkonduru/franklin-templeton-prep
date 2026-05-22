from groq import Groq
import os

from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_answer(query, contexts):

    # Merge retrieved chunks
    combined_context = "\n\n".join(contexts)

    prompt = f"""
You are a financial QA assistant.

Answer ONLY using the provided context.
Only answer if the context explicitly contains the exact information.
Do NOT infer.
Do NOT estimate.

Rules:
- Use only facts from the context
- If revenue or totals are shown, calculate carefully if needed
- Keep answers short and factual
- Do not hallucinate

If the answer is missing, say:
"The answer could not be found in the provided context."

CONTEXT:
{combined_context}

QUESTION:
{query}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=200
    )

    answer = response.choices[0].message.content

    return answer