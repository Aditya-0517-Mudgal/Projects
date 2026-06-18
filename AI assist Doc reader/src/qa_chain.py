"""
qa_chain.py
-----------
Retrieval-Augmented Generation (RAG) chain.

Flow
----
1. Retrieve top-k chunks from the vector store (semantic / keyword / hybrid).
2. Build a structured prompt with the retrieved context + role-specific instructions.
3. Call the Groq Chat Completion API (llama-3.3-70b-versatile).
4. Return {"answer": str, "sources": [...], "confidence": float, "retrieval_mode": str}
"""

import os
from typing import List, Dict, Any, Optional

from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# llama-3.3-70b-versatile is the best free-tier model on Groq:
# fast inference, strong instruction following, and large enough to
# reason over multi-document context without hallucinating.
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
# 12 000 chars ≈ 3 000 tokens — comfortably under LLaMA’s 8k context window
# while leaving room for the system prompt and the generated answer.
MAX_CONTEXT_CHARS = 12000

# ---------------------------------------------------------------------------
# 2. Role-based system prompts
# ---------------------------------------------------------------------------

# Role-specific prompts exist so the same retrieved context can be framed
# differently depending on who is asking:
# - general: neutral, factual
# - pm: structured with headings, focused on requirements and trade-offs
# - sales: concise and benefit-oriented
# All three explicitly forbid the model from going outside the context,
# which prevents hallucination on topics not covered by the uploaded documents.
SYSTEM_PROMPTS: Dict[str, str] = {
    "general": (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided document context. If the answer is not contained in the "
        "context, say: 'I could not find an answer in the provided documents.'"
    ),
    "pm": (
        "You are an assistant for a Product Manager. Answer questions strictly "
        "based on the provided document context, focusing on: features, requirements, "
        "timelines, trade-offs, user needs, and technical feasibility. Structure "
        "answers with clear headings and bullet points where appropriate. "
        "If the answer is not in the context, say so."
    ),
    "sales": (
        "You are an assistant for a Sales professional. Answer questions strictly "
        "based on the provided document context, highlighting: key benefits, value "
        "propositions, differentiators, customer pain points addressed, and "
        "competitive advantages. Keep language concise and persuasive. "
        "If the answer is not in the context, say so."
    ),
}


# ---------------------------------------------------------------------------
# 3. Private helpers
# ---------------------------------------------------------------------------

def _normalize_retrieval_scores(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-scale raw FAISS L2 distances *within the retrieved set* so the best
    match → ~90% and the worst → ~50%.

    This avoids penalising short/conversational queries that have inherently
    low absolute cosine similarity with dense document text. The score now
    reflects how relevant a chunk is *relative to the other retrieved chunks*,
    not an absolute semantic distance.
    """
    if not chunks:
        return chunks

    raw = [c.get("score", float("inf")) for c in chunks]
    finite = [s for s in raw if s != float("inf")]
    if not finite:
        return chunks

    min_d = min(finite)
    max_d = max(finite)
    spread = max_d - min_d

    for chunk in chunks:
        d = chunk.get("score", float("inf"))
        if d == float("inf"):
            chunk["score"] = 50.0  # fallback for chunks with no score
        elif spread > 0:
            # Invert: lower L2 distance = better match = higher score.
            # We map to [50%, 90%] rather than [0%, 100%] because even the
            # worst retrieved chunk still had some relevance — it was chosen
            # by the retrieval algorithm. Showing 0% would be misleading.
            # `relative` is in [0, 1]: highest for the closest chunk, lowest for the farthest.
            relative = 1.0 - (d - min_d) / spread
            chunk["score"] = round(50.0 + relative * 40.0, 1)
        else:
            # All chunks have identical L2 distances — treat them as equally relevant.
            chunk["score"] = 75.0

    return chunks


def _l2_to_confidence(score: float) -> float:
    """Return the pre-normalised query-match score (already in 0–100 range)."""
    return round(score, 1)


def _call_groq(api_key: str, user_message: str, system_prompt: Optional[str] = None) -> str:
    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError("groq package is required. Run: pip install groq") from e

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt or SYSTEM_PROMPTS["general"]},
            {"role": "user", "content": user_message},
        ],
        # temperature=0.2: low randomness keeps the model close to the provided
        # context and reduces hallucination. We don't use 0.0 because a tiny
        # amount of variation makes answers read more naturally.
        temperature=0.2,
        # max_tokens caps response length to prevent unexpectedly long completions.
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def _retrieve_chunks(
    question: str, store: VectorStore, k: int, retrieval_mode: str
) -> List[Dict[str, Any]]:
    """Dispatch to the right retrieval method based on the mode."""
    if retrieval_mode == "keyword":
        return store.search_keyword(question, k=k)
    if retrieval_mode == "hybrid":
        return store.search_hybrid(question, k=k)
    # Semantic (default). When multiple documents are indexed, use diverse
    # search so every uploaded file contributes to the answer context —
    # otherwise a large document can crowd out smaller ones entirely.
    if len(store.unique_sources) > 1:
        return store.search_diverse(question, k_per_source=3)
    return store.search(question, k=k)


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Assemble retrieved chunks into a prompt context string, capped at MAX_CONTEXT_CHARS."""
    # Sort best-scoring chunks first so that if we hit the character limit,
    # we cut the least relevant chunks rather than the most relevant ones.
    ranked = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    parts: List[str] = []
    total_chars = 0
    for chunk in ranked:
        # Label each snippet with its source so the LLM can cite it and so
        # the model stays grounded — it knows where each piece of text came from.
        snippet = (
            f"[Source: {chunk['source']} | Page: {chunk['page']}]\n"
            f"{chunk['text']}"
        )
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:  # stop before prompt gets too long
            break
        parts.append(snippet)
        total_chars += len(snippet)
    # "---" separators help the LLM treat each snippet as a distinct passage
    # rather than one continuous block of text.
    return "\n\n---\n\n".join(parts)


def _build_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate chunks by chunk_id, keeping the highest-scoring occurrence of each.

    We dedupe by chunk_id (not by (source, page)) because non-PDF formats —
    DOCX, MD, TXT — produce many chunks that all share page=1. Keying by
    (source, page) would collapse all of them into a single Sources entry and
    skew the overall confidence average toward whichever chunk happened to
    score highest. chunk_id is unique per chunk regardless of file type.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        # Fall back to the chunk's text identity if chunk_id is missing,
        # so legacy/manually-built chunks still get a reasonable dedup key.
        key = chunk.get("chunk_id") or f"{chunk['source']}__p{chunk['page']}__{id(chunk)}"
        score = chunk.get("score", 0.0)
        if key not in seen or score > seen[key]["score"]:
            seen[key] = {
                "score": score,
                "text": chunk.get("text", ""),
                "source": chunk["source"],
                "page": chunk["page"],
            }

    return [
        {
            "source": meta["source"],
            "page": meta["page"],
            "confidence": _l2_to_confidence(meta["score"]),
            "chunk_text": meta["text"],
        }
        for meta in seen.values()
    ]


def _empty_result(message: str, retrieval_mode: str) -> Dict[str, Any]:
    """Build a 'no answer' response in the standard return shape."""
    return {
        "answer": message,
        "sources": [],
        "confidence": 0.0,
        "retrieval_mode": retrieval_mode,
    }


# ---------------------------------------------------------------------------
# 4. Public entry point
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    store: VectorStore,
    groq_api_key: Optional[str] = None,
    k: int = TOP_K,
    retrieval_mode: str = "semantic",
    role: str = "general",
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks and generate an answer using Groq.

    Parameters
    ----------
    retrieval_mode : "semantic" | "keyword" | "hybrid"
    role           : "general" | "pm" | "sales"

    Returns
    -------
    {
        "answer": str,
        "sources": [{"source": str, "page": int, "confidence": float, "chunk_text": str}],
        "confidence": float,
        "retrieval_mode": str,
    }
    """
    # Lower-case so callers don't need to worry about casing.
    retrieval_mode = retrieval_mode.lower()

    # 1. Retrieve relevant chunks based on the chosen retrieval strategy.
    chunks = _retrieve_chunks(question, store, k, retrieval_mode)
    if not chunks:
        return _empty_result("No relevant content found in the indexed documents.", retrieval_mode)

    # 2. Validate API key before doing any prompt building or LLM call.
    api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return _empty_result(
            "GROQ_API_KEY is not set. Please configure it in the .env file or sidebar.",
            retrieval_mode,
        )

    # 3. Build the user message: context snippets + the question.
    context = _build_context(chunks)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context above:"
    )

    # 4. Call Groq with the role-specific system prompt.
    system_prompt = SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["general"])
    answer_text = _call_groq(api_key, user_message, system_prompt)

    # 5. Normalise raw retrieval scores into a human-readable 50–90% range
    # so the UI can display meaningful confidence values.
    # We do this AFTER calling the LLM (not before) to avoid the overhead
    # affecting latency — score normalisation is cheap and doesn't affect retrieval.
    chunks = _normalize_retrieval_scores(chunks)
    sources = _build_sources(chunks)

    # Overall confidence = average of per-source confidence scores.
    overall_confidence = (
        sum(s["confidence"] for s in sources) / len(sources) if sources else 0.0
    )

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": overall_confidence,
        "retrieval_mode": retrieval_mode,
    }