"""
Streamlit UI
Run: streamlit run ui/app.py
Expects the FastAPI server running at localhost:8000
"""

import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Financial Document Intelligence",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Financial RAG")
    st.caption("Powered by Groq · BGE Embeddings · Cohere Rerank")
    st.divider()

    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.ok:
            st.success("API online")
            data = r.json()
            st.caption(f"Model: {data.get('model', 'unknown')}")
        else:
            st.error("API error")
    except Exception:
        st.error("API offline — start with:\nuvicorn api.main:app --reload")

    # Stats
    try:
        r = requests.get(f"{API_URL}/stats", timeout=3)
        if r.ok:
            s = r.json()
            st.metric("Vectors indexed", s.get("total_vectors", 0))
    except Exception:
        pass

    st.divider()
    st.markdown("**Example questions**")
    example_qs = [
        "What was Apple's total revenue in the most recent fiscal year?",
        "How did Microsoft's cloud revenue grow year over year?",
        "What are the main risk factors Tesla discloses in its 10-K?",
        "Compare Apple and Microsoft's operating margins.",
        "What does Apple say about its competition in the 10-K?",
    ]
    for q in example_qs:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    stream_mode = st.toggle("Streaming mode", value=False)
    st.caption("Stream: see tokens as they arrive. Off: faster full response.")

# ── Main ─────────────────────────────────────────────────────────────────────

st.title("Financial Document Q&A")
st.caption("Ask any question about the 10-K filings in your database.")

# Pre-fill from sidebar button click
prefill = st.session_state.pop("prefill", "")

question = st.text_area(
    "Your question",
    value=prefill,
    height=80,
    placeholder="e.g. What was Apple's net income in FY2023?",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)
with col2:
    if question:
        st.caption(f"{len(question)} characters")

# ── Answer rendering ──────────────────────────────────────────────────────────

CONFIDENCE_COLORS = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}


def render_sources(sources: list[dict]):
    if not sources:
        return
    st.subheader("Sources", divider="gray")
    for i, src in enumerate(sources, 1):
        with st.expander(f"[{i}] {src['file']}  —  page {src['page']}  |  {src['company']} {src['fiscal_year']}"):
            st.caption(src["excerpt"])


def render_confidence(confidence: str):
    color = CONFIDENCE_COLORS.get(confidence, "gray")
    st.markdown(
        f'<span style="background-color:{"#d4edda" if color=="green" else "#fff3cd" if color=="orange" else "#f8d7da"};'
        f'color:{"#155724" if color=="green" else "#856404" if color=="orange" else "#721c24"};'
        f'padding:4px 12px;border-radius:12px;font-size:13px;font-weight:500">'
        f'Confidence: {confidence}</span>',
        unsafe_allow_html=True,
    )


if submit and question.strip():
    if stream_mode:
        # ── Streaming mode
        st.subheader("Answer")
        answer_box = st.empty()
        full_answer = ""
        try:
            with requests.post(
                f"{API_URL}/query/stream",
                json={"question": question},
                stream=True,
                timeout=60,
            ) as r:
                for line in r.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                break
                            import json
                            data = json.loads(payload)
                            if "token" in data:
                                full_answer += data["token"]
                                answer_box.markdown((full_answer + "▌").replace("$", r"\$").replace("_", r"\_"))
            answer_box.markdown(full_answer.replace("$", r"\$").replace("_", r"\_"))
        except Exception as e:
            st.error(f"Stream error: {e}")

    else:
        # ── Full response mode
        with st.spinner("Searching documents and generating answer…"):
            start = time.time()
            try:
                r = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=60,
                )
                elapsed = time.time() - start

                if r.ok:
                    data = r.json()

                    # Confidence badge
                    col_conf, col_time = st.columns([3, 1])
                    with col_conf:
                        render_confidence(data["confidence"])
                    with col_time:
                        st.caption(f"{elapsed:.1f}s · {data['source_count']} sources")

                    # Answer — escape $ and _ so Streamlit doesn't render
                    # citation filenames like AAPL_2025... as LaTeX math
                    st.subheader("Answer")
                    safe = data["answer"].replace("$", r"\$").replace("_", r"\_")
                    st.markdown(safe)

                    # Sources
                    render_sources(data["sources"])

                else:
                    detail = r.json().get("detail", r.text)
                    if "ingest" in detail.lower() or "chroma" in detail.lower():
                        st.error("No documents indexed yet.")
                        st.info("Run ingestion first:\n```\npython ingest.py\n```")
                    else:
                        st.error(f"API error: {detail}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server.")
                st.code("uvicorn api.main:app --reload --port 8000")
            except Exception as e:
                st.error(f"Error: {e}")

elif submit:
    st.warning("Please enter a question.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with LangChain LCEL · Groq (Llama 3.3 70B) · "
    "BGE-large-en-v1.5 · ChromaDB · Cohere Rerank · FastAPI · Streamlit"
)
