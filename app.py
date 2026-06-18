"""
app.py
------
Streamlit Document Q&A application.

Layout
------
Sidebar
  ├── File uploader (PDF / DOCX / MD / TXT)
  ├── Groq API key input (prefilled from env)
  └── "Build Index" button + status

Main area
  ├── Chat history (user + assistant turns)
  ├── Text input for questions
  └── Expandable "Sources" section per answer
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path fix so `src` package resolves when run from project root
# ---------------------------------------------------------------------------
# sys.path.insert ensures Python finds the `src` package when the app is
# launched from the project root with `streamlit run app.py`. Without this,
# `from src.document_loader import ...` would raise ModuleNotFoundError.
sys.path.insert(0, os.path.dirname(__file__))

from src.document_loader import load_from_bytes
from src.chunker import chunk_documents
from src.vector_store import VectorStore
from src.qa_chain import answer_question

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Display labels for the retrieval modes. Two variants:
# - VERBOSE: shown in the sidebar selector (includes the algorithm name)
# - SHORT:   shown next to assistant turns in the chat history
#
# What each mode does under the hood:
# - "semantic" → embed query → FAISS IndexFlatL2.search() → return nearest chunks.
#                "Semantic" is the *technique* (matching by meaning, not exact words);
#                FAISS is just the library that makes the vector lookup fast.
#                Swapping FAISS for Pinecone, Qdrant, Chroma, or even raw NumPy
#                would still be "semantic" search.
# - "keyword"  → tokenise query → BM25Okapi term-frequency scoring (no FAISS involved).
# - "hybrid"   → run both of the above, fuse their rankings with Reciprocal Rank Fusion.
RETRIEVAL_MODE_LABELS_VERBOSE = {
    "semantic": "🔍 Semantic",
    "keyword":  "🔑 Keyword (BM25)",
    "hybrid":   "⚡ Hybrid (RRF)",
}
RETRIEVAL_MODE_LABELS_SHORT = {
    "semantic": "🔍 Semantic",
    "keyword":  "🔑 Keyword",
    "hybrid":   "⚡ Hybrid",
}


def _confidence_badge(confidence: float) -> str:
    """Return an emoji label for a 0–100 query match score.

    Scores are relative within the retrieved set: best chunk → ~90%, worst → ~50%.
    """
    # Three tiers give an at-a-glance signal without the user needing to read the number.
    # Thresholds (75 / 60) split the [50–90%] range into equal thirds.
    # Green = strong match, yellow = moderate, red = weakest in this retrieved set
    if confidence >= 75:
        return f"🟢 {confidence:.1f}%"
    elif confidence >= 60:
        return f"🟡 {confidence:.1f}%"
    else:
        return f"🔴 {confidence:.1f}%"

# ---------------------------------------------------------------------------
# Load environment variables from .env (if present)
# ---------------------------------------------------------------------------
# load_dotenv reads a .env file in the project root (if it exists) and injects
# its contents as environment variables. This lets developers set GROQ_API_KEY
# locally without hardcoding it or exporting it in every terminal session.
load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)

# Hide Streamlit's default footer and main menu — they add clutter and expose
# options (like "Rerun") that aren't useful here.
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
# st.session_state persists values across Streamlit reruns (triggered by any
# user interaction). We guard each key with `not in` so it's initialised only
# once — on first load — and not reset every time the user clicks something.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # [{"role": "user"|"assistant", "content": str, "sources": list}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # holds the built FAISS index (VectorStore | None)

if "index_info" not in st.session_state:
    st.session_state.index_info = ""  # human-readable summary shown in the sidebar


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Document Q&A")
    st.markdown("Upload documents, build the index, then ask questions.")

    # --- Groq API key (read from environment only) ---
    # We intentionally don't render a text input for the key — showing it in
    # the UI risks it appearing in screen recordings or shared screenshots.
    # Users set it once in .env or as an OS environment variable.
    groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.divider()

    # --- File uploader ---
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, Markdown, plain text",
    )

    # --- Build index button ---
    build_clicked = st.button("\U0001F4DA Analyze Documents", use_container_width=True, type="primary")

    if build_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Loading and chunking documents…"):
                all_pages = []
                failed = []
                for uf in uploaded_files:
                    try:
                        # load_from_bytes detects the file type from content (not
                        # the extension), writes to a temp file, parses it, and cleans up.
                        pages = load_from_bytes(uf.read(), uf.name)
                        all_pages.extend(pages)
                    except Exception as exc:
                        # Collect failures instead of stopping — one bad file
                        # shouldn't block the rest from being indexed.
                        failed.append(f"{uf.name}: {exc}")

                if failed:
                    st.error("Failed to load:\n" + "\n".join(failed))

                # Split pages into overlapping chunks before embedding.
                # Smaller chunks give retrieval more precise matches;
                # overlap prevents answers from being lost at boundaries.
                chunks = chunk_documents(all_pages)

            if chunks:
                with st.spinner(f"Embedding {len(chunks)} chunks…"):
                    store = VectorStore()
                    # build() runs the embedding model over all chunks and
                    # constructs both the FAISS (semantic) and BM25 (keyword) indices.
                    store.build(chunks)
                    st.session_state.vector_store = store
                    st.session_state.index_info = (
                        f"{len(uploaded_files)} file(s) · {len(all_pages)} page(s) · {len(chunks)} chunks"
                    )
                    # Reset chat history so previous answers (which reference
                    # the old index) don't confuse the user after a new upload.
                    st.session_state.chat_history = []
                st.success("Index built successfully!")
            else:
                st.error("No text could be extracted from the uploaded files.")

    # --- Index status ---
    if st.session_state.index_info:
        st.info(f"**Active index:** {st.session_state.index_info}")

    st.divider()

    # --- Retrieval mode ---
    st.subheader("⚙️ Settings")
    retrieval_mode = st.radio(
        "Retrieval Algorithm",
        options=["semantic", "keyword", "hybrid"],
        format_func=lambda x: RETRIEVAL_MODE_LABELS_VERBOSE[x],
        help="Semantic: dense vector search  |  Keyword: BM25 term matching  |  Hybrid: fuses both via RRF",
        horizontal=True,
    )

    # --- Role selector ---
    role = st.selectbox(
        "Answer Style",
        options=["general", "pm", "sales"],
        format_func=lambda x: {"general": "💬 General", "pm": "📋 Product Manager", "sales": "💼 Sales"}[x],
        help="Tailors the LLM's answer framing to your role",
    )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("💬 Chat with your Documents")

# --- Display chat history ---
for turn_idx, turn in enumerate(st.session_state.chat_history):
    turn_role = turn["role"]
    content = turn["content"]
    sources = turn.get("sources", [])

    with st.chat_message(turn_role):
        st.markdown(content)
        if turn_role == "assistant" and turn.get("retrieval_mode"):
            mode_label = RETRIEVAL_MODE_LABELS_SHORT.get(turn["retrieval_mode"], turn["retrieval_mode"])
            st.caption(f"Retrieved via {mode_label}")
        if sources:
            confidence = turn.get("confidence")
            label = f"📚 Sources ({len(sources)})"
            if confidence is not None:
                label += f"  ·  {_confidence_badge(confidence)}"
            with st.expander(label):
                if confidence is not None:
                    st.progress(int(confidence), text=f"Query match: **{confidence:.1f}%**")
                    st.caption("ℹ️ How closely this source's content matched your query, relative to the other retrieved chunks.")
                for src_idx, src in enumerate(sources):
                    src_conf = src.get("confidence")
                    conf_str = f"  `{src_conf:.1f}%`" if src_conf is not None else ""
                    st.markdown(f"- **{src['source']}** — page {src['page']}{conf_str}")
                    if src.get("chunk_text"):
                        st.text_area("📄 Matched text", src["chunk_text"], height=120, disabled=True, label_visibility="visible", key=f"hist_{turn_idx}_{src_idx}")

# --- Question input ---
question = st.chat_input("Ask a question about your documents…")

if question:
    # Validate preconditions
    if st.session_state.vector_store is None:
        st.warning("Please upload documents and click **Analyze Documents** first.")
        st.stop()

    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    # Append user turn
    st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = answer_question(
                question=question,
                store=st.session_state.vector_store,
                groq_api_key=groq_api_key,
                retrieval_mode=retrieval_mode,
                role=role,
            )

        answer = result["answer"]
        sources = result["sources"]
        confidence = result.get("confidence")
        used_mode = result.get("retrieval_mode", retrieval_mode)

        st.markdown(answer)
        mode_label = RETRIEVAL_MODE_LABELS_SHORT.get(used_mode, used_mode)
        st.caption(f"Retrieved via {mode_label}")
        if sources:
            label = f"📚 Sources ({len(sources)})"
            if confidence is not None:
                label += f"  ·  {_confidence_badge(confidence)}"
            with st.expander(label):
                if confidence is not None:
                    st.progress(int(confidence), text=f"Query match: **{confidence:.1f}%**")
                    st.caption("ℹ️ How closely this source's content matched your query, relative to the other retrieved chunks.")
                for src_idx, src in enumerate(sources):
                    src_conf = src.get("confidence")
                    conf_str = f"  `{src_conf:.1f}%`" if src_conf is not None else ""
                    st.markdown(f"- **{src['source']}** — page {src['page']}{conf_str}")
                    if src.get("chunk_text"):
                        st.text_area("📄 Matched text", src["chunk_text"], height=120, disabled=True, label_visibility="visible", key=f"live_{src_idx}")

    # Append assistant turn
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources, "confidence": confidence, "retrieval_mode": used_mode}
    )