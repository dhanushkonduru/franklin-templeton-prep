from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext
)

from llama_index.core.node_parser import (
    SentenceSplitter
)

from llama_index.vector_stores.qdrant import (
    QdrantVectorStore
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

from qdrant_client import QdrantClient
from pathlib import Path
import re


def _company_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()

    company_map = {
        "blackrock": "BlackRock",
        "goldman": "Goldman Sachs",
        "jpmorgan": "JPMorgan Chase",
        "morgan stanley": "Morgan Stanley",
    }

    return company_map.get(stem, stem.replace("_", " ").title())


def _report_year_from_filename(filename: str):
    match = re.search(r"(20\d{2})", filename)
    return int(match.group(1)) if match else None


# Load documents
documents = SimpleDirectoryReader(
    "data/processed"
).load_data()

for document in documents:
    file_name = document.metadata.get("file_name", "")
    source_filename = Path(file_name).name if file_name else ""

    document.metadata.update({
        "source_filename": source_filename,
        "company_name": _company_name_from_filename(source_filename),
        "report_year": _report_year_from_filename(source_filename),
    })


# Better chunking
parser = SentenceSplitter(
    chunk_size=700,
    chunk_overlap=100
)

nodes = parser.get_nodes_from_documents(
    documents
)


# Embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# Qdrant local database
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
    vector_store=vector_store
)


# Build index
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)


# Persist index
index.storage_context.persist(
    persist_dir="data/index"
)

print("Index created successfully")