from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import generate_answer
from rag.verifier import verify_answer


def run_rag(query):

    retrieved_docs = retrieve(query)

    reranked_docs = rerank(
        query,
        retrieved_docs
    )

    contexts = []

    for doc in reranked_docs[:3]:
        contexts.append(doc["text"])

    answer = generate_answer(
        query=query,
        contexts=contexts
    )

    # Use only the first context (truncated) for faster, stable NLI verification
    verification_context = contexts[0][:1000] if contexts else ""

    verification = verify_answer(
        verification_context,
        answer
    )

    if (
    verification["label"].lower() == "contradiction"
    and verification["score"] > 0.80
    ):
        answer = "Potential hallucination detected."

    return {
    "question": query,
    "answer": answer,
    "contexts": contexts,
    "verification": verification
}