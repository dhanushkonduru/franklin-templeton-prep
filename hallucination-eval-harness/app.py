import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from rag.pipeline import run_rag

st.set_page_config(
    page_title="Hallucination Evaluation Harness",
    layout="wide"
)

st.title("Financial RAG Hallucination Detection")

query = st.text_input(
    "Ask a financial question"
)

if st.button("Run RAG"):

    with st.spinner("Running pipeline..."):

        try:
            result = run_rag(query)

        except Exception as e:
            st.error(str(e))
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Verification")
    st.json(result["verification"])

    st.subheader("Retrieved Contexts")

    for i, ctx in enumerate(result["contexts"]):

        st.markdown(f"### Context {i+1}")

        st.write(ctx[:1500])