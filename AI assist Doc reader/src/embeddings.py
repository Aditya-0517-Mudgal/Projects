"""
embeddings.py
-------------
Wraps sentence-transformers to produce dense vector embeddings.
The model is held in a module-level singleton so it loads from disk only
once per process (across Streamlit reruns, tests, or CLI scripts).
"""

from typing import List
import numpy as np


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# all-MiniLM-L6-v2 is a good default for document Q&A:
# - Small (80 MB) and fast on CPU — no GPU required
# - Produces 384-dimensional vectors, compact enough for FAISS IndexFlatL2
# - Pre-trained on MS MARCO passage-retrieval data, so it understands
#   query-document similarity out of the box without any fine-tuning
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# 2. Private helpers (model loading + singleton)
# ---------------------------------------------------------------------------

def _load_model():
    """Load (or return cached) SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Run: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(MODEL_NAME)


# A module-level singleton avoids reloading the model from disk on every call.
# Loading a SentenceTransformer model takes ~1–2 seconds; without caching,
# every embed call during indexing (hundreds of chunks) would reload it.
# We don't use @st.cache_resource so the model also works outside Streamlit
# (e.g. tests or CLI scripts).
_model = None


def get_model():
    global _model
    if _model is None:
        _model = _load_model()  # lazy-load: only downloads/loads on first call
    return _model


# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into a float32 numpy array of shape (N, D).
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # sentence-transformers may return float64 depending on the platform.
    # FAISS's IndexFlatL2 requires float32, so we always cast here to avoid
    # a silent type mismatch that would cause index.add() to fail.
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string → shape (1, D) float32 array.
    """
    # Wrap in a list to reuse embed_texts — one encoding path for both chunks
    # and queries. FAISS index.search() expects shape (1, D), which is what
    # embed_texts([query]) returns.
    return embed_texts([query])