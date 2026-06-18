"""
vector_store.py
---------------
FAISS-backed vector store.

Responsibilities
----------------
* build(chunks)  – embed all chunks and create an IndexFlatL2 index
* search(query, k=5) – embed the query and return top-k chunk dicts
"""

from typing import List, Dict, Any

import numpy as np

from src.embeddings import embed_texts, embed_query


# ---------------------------------------------------------------------------
# VectorStore class
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self):
        self._index = None          # faiss.Index — holds all chunk vectors
        # _metadata[i] mirrors index vector i — they must stay in sync.
        # When FAISS returns index position 42, we look up _metadata[42] for
        # the original text, source filename, and page number.
        self._metadata: List[Dict[str, Any]] = []
        self._bm25 = None           # BM25Okapi — parallel keyword index over the same corpus

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed all chunks and populate the FAISS + BM25 indices."""
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss-cpu is required. Run: pip install faiss-cpu") from e
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank-bm25 is required. Run: pip install rank-bm25") from e

        if not chunks:
            raise ValueError("chunks list is empty – nothing to index.")

        texts = [c["text"] for c in chunks]
        embeddings: np.ndarray = embed_texts(texts)  # shape: (N, D) float32

        # IndexFlatL2: exact brute-force L2 distance search.
        # We use this (not an approximate index like HNSW) because our index
        # sizes are small enough that exact search is fast, and we never want
        # to miss a relevant chunk due to approximation error.
        self._index = faiss.IndexFlatL2(embeddings.shape[1])  # 384 for all-MiniLM-L6-v2
        self._index.add(embeddings)  # vectors are stored in insertion order

        # BM25 needs a tokenised corpus. Simple whitespace splitting is fine here —
        # we don't need stemming or stop-word removal for basic keyword matching.
        # BM25Okapi keeps its own internal reference to the corpus, so we don't
        # store it on self.
        self._bm25 = BM25Okapi([t.lower().split() for t in texts])

        # Store full metadata per chunk, including the text itself.
        # We keep text in metadata (not just in the FAISS index) because FAISS
        # only stores vectors — it has no way to return the original text.
        self._metadata = [
            {
                "text": c["text"],
                "source": c["source"],
                "page": c["page"],
                "chunk_id": c.get("chunk_id", ""),
            }
            for c in chunks
        ]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k most similar chunks for *query*."""
        if self._index is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        query_vec = embed_query(query)  # shape: (1, D)
        distances, indices = self._index.search(query_vec, k)
        # distances[0]: L2 distances in ascending order (lower = more similar)
        # indices[0]:   corresponding positions in the FAISS index

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 for padding when the index has fewer than k vectors.
            # Skipping these prevents an IndexError on _metadata[-1].
            if idx == -1:
                continue
            chunk = dict(self._metadata[idx])  # copy so callers can safely mutate it
            chunk["score"] = float(dist)       # raw L2 distance; normalised later in qa_chain
            results.append(chunk)

        return results

    def search_diverse(self, query: str, k_per_source: int = 3) -> List[Dict[str, Any]]:
        """Return top-k chunks *per unique source document*.

        Guarantees representation from every uploaded file, even if one
        document dominates the index by size.
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        sources = self.unique_sources
        if len(sources) <= 1:
            # Only one document — no diversity problem to solve, plain search is fine
            return self.search(query, k=k_per_source * 2)

        query_vec = embed_query(query)
        # Over-fetch candidates so we have enough results to fill k_per_source
        # slots for every source, even if one document dominates by chunk count.
        k_total = min(k_per_source * len(sources) * 4, self._index.ntotal)
        distances, indices = self._index.search(query_vec, k_total)

        # Bucket top results by source, capping at k_per_source per document.
        # Without this, a large document could fill all k slots and leave
        # smaller documents completely unrepresented in the answer context.
        per_source: Dict[str, List[Dict[str, Any]]] = {s: [] for s in sources}
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self._metadata[idx])
            chunk["score"] = float(dist)
            src = chunk["source"]
            if src in per_source and len(per_source[src]) < k_per_source:
                per_source[src].append(chunk)

        # Interleave results round-robin: 1st best from doc A, 1st best from doc B,
        # then 2nd best from doc A, 2nd best from doc B, etc.
        # This ensures the LLM sees context from all documents before hitting
        # the MAX_CONTEXT_CHARS limit in qa_chain.
        results: List[Dict[str, Any]] = []
        max_len = max(len(v) for v in per_source.values()) if per_source else 0
        for i in range(max_len):
            for src in sources:
                if i < len(per_source[src]):
                    results.append(per_source[src][i])

        return results

    def search_keyword(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword search — returns top-k chunks by term frequency."""
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build() first.")

        tokens = query.lower().split()  # same tokenisation used when building the corpus
        scores = self._bm25.get_scores(tokens)  # one score per chunk in the corpus

        # Sort all (index, score) pairs descending and take the top k.
        # We can't ask BM25 for top-k directly; we have to score everything.
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            # score=0 means none of the query tokens appeared in this chunk at all.
            # Including zero-score chunks would add noise with no signal.
            if score == 0:
                continue
            chunk = dict(self._metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def search_hybrid(self, query: str, k: int = 5, semantic_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Hybrid search using Reciprocal Rank Fusion (RRF) of semantic + BM25 results.

        RRF formula: score(d) = Σ 1 / (k_rrf + rank(d))
        Combines rankings from both retrieval methods without needing to normalise scores.
        """
        if self._index is None or self._bm25 is None:
            raise RuntimeError("Index not built. Call build() first.")

        # K_RRF=60 is the standard constant from the original RRF paper
        # (Cormack et al., 2009). A higher value dampens rank differences;
        # 60 is widely used in production retrieval systems.
        K_RRF = 60
        # Fetch 4x more candidates than we'll return so RRF has a rich pool
        # to fuse from — both methods may rank the same chunk differently.
        candidate_k = min(k * 4, self._index.ntotal)

        # --- Semantic ranking: nearest neighbours by L2 distance ---
        query_vec = embed_query(query)
        # We only need the ranking order for RRF, not the raw L2 distances —
        # so we discard the distances array with `_`.
        _, indices = self._index.search(query_vec, candidate_k)
        # Map chunk_index → rank (0 = best match)
        semantic_ranks: Dict[int, int] = {
            int(idx): rank
            for rank, idx in enumerate(indices[0])
            if idx != -1
        }

        # --- Keyword ranking: BM25 term frequency scores ---
        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)
        keyword_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        keyword_ranks: Dict[int, int] = {idx: rank for rank, idx in enumerate(keyword_ranked)}

        # --- RRF fusion ---
        # For each chunk seen in either ranking, compute:
        #   RRF_score = semantic_weight / (K_RRF + sem_rank)
        #             + (1 - semantic_weight) / (K_RRF + kw_rank)
        # Chunks missing from one method get a penalty rank of candidate_k,
        # so they're not completely excluded but score lower than present ones.
        all_ids = set(semantic_ranks) | set(keyword_ranks)
        rrf_scores: Dict[int, float] = {}
        for idx in all_ids:
            sem_score = semantic_weight / (K_RRF + semantic_ranks.get(idx, candidate_k))
            kw_score = (1 - semantic_weight) / (K_RRF + keyword_ranks.get(idx, len(bm25_scores)))
            rrf_scores[idx] = sem_score + kw_score

        top_k = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:k]
        results: List[Dict[str, Any]] = []
        for idx in top_k:
            chunk = dict(self._metadata[idx])
            chunk["score"] = rrf_scores[idx]  # fused RRF score (not comparable to raw L2 or BM25)
            results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def unique_sources(self) -> List[str]:
        """Ordered list of unique source filenames present in the index."""
        seen: set = set()
        sources: List[str] = []
        for m in self._metadata:
            s = m["source"]
            if s not in seen:
                seen.add(s)
                sources.append(s)
        return sources