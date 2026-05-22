import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_nli_model():
    return pipeline(
        "text-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device=-1
    )


nli_model = load_nli_model()


def verify_answer(context, answer):
    contexts = context if isinstance(context, list) else [context]
    combined_context = " ".join(contexts[:2])[:900]

    result = nli_model(
        {
            "text": combined_context,
            "text_pair": answer
        }
    )

    return {
        "label": result["label"],
        "score": result["score"]
    }