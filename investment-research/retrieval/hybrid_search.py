from retrieval.vectorstore import collection
from rank_bm25 import BM25Okapi

def dense_search(query_embedding):

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    return results["documents"][0]

def bm25_search(chunks, query):

    tokenized_chunks = [
        chunk.split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    scores = bm25.get_scores(
        query.split()
    )

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in ranked[:5]]