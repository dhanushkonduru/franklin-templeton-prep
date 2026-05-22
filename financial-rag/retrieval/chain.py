"""
RAG Chain (LCEL)
User question → retrieval → citation prompt → Groq LLM → structured answer
Returns answer text + structured source list with page numbers.
"""

import re
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema import Document
from loguru import logger

from config import get_settings
from retrieval.retriever import build_retriever

settings = get_settings()

# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior financial analyst assistant. You have been given excerpts from SEC 10-K filings, earnings call transcripts, and broker research reports.

RULES:
1. Answer ONLY from the provided context. Do not use prior knowledge.
2. For EVERY factual claim, add a citation in this exact format: [filename, p.PAGE] — use the exact filename from the context header, nothing else.
3. If the answer involves numbers, quote them exactly as they appear in the context.
4. At the end of your answer, write a CONFIDENCE line:
   CONFIDENCE: HIGH | MEDIUM | LOW
   - HIGH = answer is directly stated in context
   - MEDIUM = answer requires some inference from context
   - LOW = context is tangential, answer may be incomplete
5. If the context does not contain enough information, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
   Then explain what information is missing.

Do not fabricate numbers, dates, or facts."""

USER_PROMPT = """Context documents:
{context}

---
Question: {question}

Answer (with citations and confidence):"""

CITATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])


# ── Output structures ────────────────────────────────────────────────────────

@dataclass
class Source:
    file: str
    page: int
    excerpt: str
    company: str = ""
    fiscal_year: str = ""


@dataclass
class RAGResult:
    question: str
    answer: str
    confidence: str
    sources: list[Source] = field(default_factory=list)
    raw_context: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    """Format retrieved docs into the context string for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        company = meta.get("company", "")
        year = meta.get("fiscal_year", "")
        header = f"[{i}] {src} | p.{page} | {company} {year}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_confidence(answer: str) -> tuple[str, str]:
    """Extract confidence level from the LLM answer and return (clean_answer, confidence)."""
    match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", answer, re.IGNORECASE)
    if match:
        confidence = match.group(1).upper()
        clean = answer[:match.start()].strip()
        return clean, confidence
    return answer.strip(), "MEDIUM"


def _extract_sources(docs: list[Document]) -> list[Source]:
    return [
        Source(
            file=d.metadata.get("source", "unknown"),
            page=int(d.metadata.get("page", 0)),
            excerpt=d.page_content[:200].replace("\n", " "),
            company=d.metadata.get("company", ""),
            fiscal_year=d.metadata.get("fiscal_year", ""),
        )
        for d in docs
    ]


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm():
    llm = ChatGroq(
        model=settings.groq_model,
        temperature=0,
        api_key=settings.groq_api_key,
        max_tokens=2048,
    )
    # Optional: add Together AI fallback
    if settings.together_api_key:
        try:
            from langchain_together import ChatTogether
            fallback = ChatTogether(
                model=settings.together_model,
                together_api_key=settings.together_api_key,
                temperature=0,
            )
            return llm.with_fallbacks([fallback])
        except Exception:
            pass
    return llm


# ── Main chain ────────────────────────────────────────────────────────────────

_chain = None
_retriever = None


def get_chain():
    global _chain, _retriever
    if _chain is None:
        _retriever = build_retriever()
        llm = get_llm()

        _chain = (
            {
                "context": _retriever | RunnableLambda(_format_context),
                "question": RunnablePassthrough(),
            }
            | CITATION_PROMPT
            | llm
            | StrOutputParser()
        )
        logger.success("RAG chain initialised (LCEL)")
    return _chain, _retriever


def query(question: str) -> RAGResult:
    """Run a question through the full RAG pipeline."""
    logger.info(f"Query: {question[:80]}...")

    chain, retriever = get_chain()

    # Get retrieved docs separately for source extraction
    retrieved_docs: list[Document] = retriever.invoke(question)

    # Run the full chain
    raw_answer: str = chain.invoke(question)

    # Parse confidence
    answer, confidence = _extract_confidence(raw_answer)

    # Build sources
    sources = _extract_sources(retrieved_docs)

    result = RAGResult(
        question=question,
        answer=answer,
        confidence=confidence,
        sources=sources,
        raw_context=_format_context(retrieved_docs),
    )

    logger.info(f"Answer ready — confidence: {confidence} | sources: {len(sources)}")
    return result


def stream_query(question: str):
    """Stream answer tokens. Yields str chunks."""
    chain, _ = get_chain()
    retriever = build_retriever()
    docs = retriever.invoke(question)
    context = _format_context(docs)
    prompt_value = CITATION_PROMPT.format_messages(context=context, question=question)
    llm = get_llm()
    for chunk in llm.stream(prompt_value):
        yield chunk.content
