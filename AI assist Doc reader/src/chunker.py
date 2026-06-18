"""
chunker.py
----------
Splits raw page dicts into smaller, overlapping text chunks suitable for
embedding and retrieval.

Chunk size  : 1000 characters
Overlap     : 100 characters (10%)
"""

from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# 1000 chars ≈ 150–200 words, which sits comfortably within the 256-token
# context window of all-MiniLM-L6-v2. Going larger risks truncation inside
# the embedding model; going smaller increases the number of chunks and index size.
CHUNK_SIZE = 1000

# 10% overlap (100 chars) means consecutive chunks share a small tail/head.
# This prevents a sentence that straddles a chunk boundary from being split
# and lost. We chose 10% rather than the original 20% to reduce redundant
# content in the index while still covering most boundary cases.
CHUNK_OVERLAP = 100  # 10% of CHUNK_SIZE


# ---------------------------------------------------------------------------
# 2. Private splitter
# ---------------------------------------------------------------------------

def _split_text(text: str) -> List[str]:
    """
    Sliding-window character splitter.
    Returns at least one chunk even for very short texts.
    """
    # Short texts don't need splitting — returning them as-is avoids creating
    # a single tiny chunk that scores poorly against longer query contexts.
    if len(text) <= CHUNK_SIZE:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:  # skip chunks that are purely whitespace after stripping
            chunks.append(chunk)
        if end >= len(text):  # consumed all the text, stop
            break
        # Step forward by (CHUNK_SIZE - CHUNK_OVERLAP), not CHUNK_SIZE.
        # This places the next chunk's start CHUNK_OVERLAP chars before the end
        # of the current chunk, creating the shared overlap window.
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------

def chunk_documents(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes the output of document_loader and returns a flat list of chunk dicts.
    Each chunk dict has the same keys as the input page plus a `chunk_id` field.
    """
    chunks: List[Dict[str, Any]] = []

    for page in pages:
        text: str = page["text"]
        source: str = page["source"]
        page_num: int = page["page"]

        # Chunking is per-page (not across the whole document) so that
        # chunks never cross a page boundary. This keeps page attribution
        # accurate — every chunk knows exactly which page it came from.
        page_chunks = _split_text(text)
        for idx, chunk_text in enumerate(page_chunks):
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page_num,
                # chunk_id format: "<filename>__p<page>__c<index>"
                # e.g. "report.pdf__p3__c2" = report.pdf, page 3, 3rd chunk.
                # Used for deduplication and traceability in search results.
                "chunk_id": f"{source}__p{page_num}__c{idx}",
            })

    return chunks