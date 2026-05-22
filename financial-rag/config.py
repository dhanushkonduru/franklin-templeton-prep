from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # Fallback LLM
    together_api_key: str = Field("", env="TOGETHER_API_KEY")
    together_model: str = Field(
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", env="TOGETHER_MODEL"
    )

    # Reranker
    cohere_api_key: str = Field(..., env="COHERE_API_KEY")
    cohere_rerank_model: str = Field("rerank-english-v3.0", env="COHERE_RERANK_MODEL")

    # Embeddings (local HuggingFace — no key needed)
    embedding_model: str = Field(
        "BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL"
    )
    embedding_device: str = Field("cpu", env="EMBEDDING_DEVICE")

    # Vector DB
    chroma_persist_dir: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field("financial_docs", env="CHROMA_COLLECTION")

    # Retrieval
    retrieval_top_k: int = Field(20, env="RETRIEVAL_TOP_K")   # before rerank
    rerank_top_n: int = Field(5, env="RERANK_TOP_N")          # after rerank
    bm25_weight: float = Field(0.3, env="BM25_WEIGHT")
    dense_weight: float = Field(0.7, env="DENSE_WEIGHT")

    # Chunking
    chunk_size: int = Field(800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(100, env="CHUNK_OVERLAP")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
