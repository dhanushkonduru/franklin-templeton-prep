from tools.transcript_tools import (
    get_earnings_transcript
)

from retrieval.chunking import chunk_text

from retrieval.embeddings import (
    create_embeddings,
    model
)

from retrieval.vectorstore import (
    store_chunks
)

from retrieval.hybrid_search import (
    dense_search,
    bm25_search
)

from retrieval.reranker import rerank

from retrieval.compression import (
    compress_context
)

def retrieve_context(ticker, query):

    transcript = get_earnings_transcript(
        ticker
    )

    chunks = chunk_text(transcript)

    embeddings = create_embeddings(chunks)

    store_chunks(chunks, embeddings)

    query_embedding = model.encode(query)

    dense_results = dense_search(
        query_embedding
    )

    bm25_results = bm25_search(
        chunks,
        query
    )

    combined = (
        dense_results + bm25_results
    )

    reranked = rerank(
        query,
        combined
    )

    compressed = compress_context(
        reranked
    )

    return compressed