import os
import cohere

from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(
    os.getenv("COHERE_API_KEY")
)

def rerank(query, docs):

    response = co.rerank(
        query=query,
        documents=docs,
        top_n=3,
        model="rerank-english-v3.0"
    )

    results = []

    for item in response.results:
        results.append(
            docs[item.index]
        )

    return results