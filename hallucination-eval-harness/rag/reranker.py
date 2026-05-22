import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
from sentence_transformers import CrossEncoder


@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )


reranker = load_reranker()

def rerank(query, docs):

    pairs = []

    for doc in docs:
        pairs.append([query, doc["text"]])

    scores = reranker.predict(pairs)

    scored_docs = []

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)
        scored_docs.append(doc)

    scored_docs = sorted(
        scored_docs,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return scored_docs