from llama_index.core import (
    StorageContext,
    load_index_from_storage
)

from llama_index.vector_stores.qdrant import (
    QdrantVectorStore
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

from qdrant_client import QdrantClient


# Embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Qdrant client
client = QdrantClient(
    path="data/qdrant"
)

# Vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="financial_docs"
)

# Storage context
storage_context = StorageContext.from_defaults(
    persist_dir="data/index",
    vector_store=vector_store
)

# Load index
index = load_index_from_storage(
    storage_context=storage_context,
    embed_model=embed_model
)

# Retriever
retriever = index.as_retriever(
    similarity_top_k=3
)


def retrieve(query):

    normalized_query = query.lower()
    company_hint = "blackrock" if "blackrock" in normalized_query else None

    year_hint = None
    if "2024" in normalized_query:
        year_hint = 2024
    elif "2023" in normalized_query:
        year_hint = 2023

    nodes = retriever.retrieve(query)

    results = []

    for node in nodes:

        metadata = node.metadata or {}
        score = node.score if node.score is not None else 0.0
        boost = 0.0

        if company_hint:
            company_name = str(metadata.get("company_name", "")).lower()
            source_filename = str(metadata.get("source_filename", "")).lower()

            if company_hint in company_name or company_hint in source_filename:
                boost += 1.0

        if year_hint is not None and metadata.get("report_year") == year_hint:
            boost += 0.5

        results.append({
            "text": node.text,
            "metadata": metadata,
            "score": score + boost
        })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results