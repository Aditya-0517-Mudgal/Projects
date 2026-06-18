"""
document_loader.py
------------------
Loads PDF, DOCX, Markdown, and plain-text files and returns a list of
page/section dicts: {"text": str, "source": str, "page": int}

File type is determined from magic bytes (file content), not the filename
extension, so renamed or mislabelled files are handled correctly.
"""

import os
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# 512 bytes is the sweet spot for detection:
# - Large enough to avoid cutting mid-sequence through a multi-byte Unicode character
#   (e.g. `‑` U+2011 needs 3 bytes; an 8-byte window could land right in the middle of it,
#   making valid UTF-8 text look broken — this was the original bug that caused KB_v1.1.md to fail).
# - Small enough that we never read more than a tiny slice of the file, keeping detection fast.
_MAGIC_HEADER_SIZE = 512

# Every valid PDF file on any OS/tool starts with these exact 4 bytes.
# This is part of the PDF specification (ISO 32000), so it is a 100% reliable signal.
_PDF_MAGIC = b"%PDF"

# DOCX (and all modern Office Open XML formats) are ZIP archives, so they share
# the same 4-byte magic with .xlsx, .pptx, .jar, etc. We use the extension as a
# secondary hint to disambiguate — we still check the magic first (content-first),
# and only trust the extension when we already know we're looking at a ZIP.
_ZIP_MAGIC = b"PK\x03\x04"


# ---------------------------------------------------------------------------
# 2. Magic-byte detection helpers
# ---------------------------------------------------------------------------

def _detect_type_from_bytes(data: bytes, filename: str = "") -> str: #works on raw bytes (used by load_from_bytes)
    """Return 'pdf', 'docx', or 'text' based on the file's magic bytes.

    `filename` is used only as a tiebreaker for ZIP-based formats (DOCX vs XLSX
    etc.) — it does not override magic-byte detection for PDF or text.

    Raises ValueError if the content cannot be identified as a supported type.
    """
    # Check PDF first — it's a hard binary signature, so we can decide instantly
    # without doing any text decoding.
    if data[:4] == _PDF_MAGIC:
        return "pdf"

    # DOCX is a ZIP file. The magic bytes alone can't tell DOCX from XLSX/PPTX/etc.,
    # so we require the .docx extension as a secondary confirmation.
    # We still check the magic first so a file with a .docx extension that isn't
    # actually a ZIP (e.g. a corrupted file) gets rejected here rather than
    # crashing inside python-docx.
    if data[:4] == _ZIP_MAGIC:
        if filename.lower().endswith(".docx"):
            return "docx"
        raise ValueError(
            "File content is not a recognised type. "
            "Supported formats: PDF, DOCX, Markdown, plain text."
        )

    # For text files we can't rely on a fixed signature, so instead we try to
    # decode the sample as UTF-8 and measure how much of it is unreadable.
    # We use errors="replace" (not errors="strict") deliberately: strict would
    # raise an exception if even a single byte at the 512-byte boundary happens
    # to be the middle of a multi-byte character — giving a false rejection for
    # perfectly valid files (the original KB_v1.1.md bug). "replace" turns those
    # unreadable bytes into the Unicode replacement character \ufffd instead.
    decoded = data.decode("utf-8", errors="replace")

    # Count what fraction of the decoded characters are replacements.
    # A genuine text file might have 1-2 from a boundary cut, but a binary file
    # (e.g. an image or ZIP) will have far more — we use 10% as the cutoff.
    replacement_ratio = decoded.count("\ufffd") / max(len(decoded), 1)
    if replacement_ratio > 0.1:
        raise ValueError(
            "File content is not a recognised type. "
            "Supported formats: PDF, DOCX, Markdown, plain text."
        )
    return "text"


def _detect_type_from_path(file_path: str) -> str: #opens file, reads 512 bytes, calls above
    """Read the first bytes of a file on disk and return its detected type."""
    # Open in binary mode ("rb") — we must read raw bytes before any decoding
    # so the magic signature check works correctly regardless of platform line endings.
    with open(file_path, "rb") as fh:
        header = fh.read(_MAGIC_HEADER_SIZE)  # read only the header, not the whole file
    # Pass the filename so _detect_type_from_bytes can use the .docx extension
    # as a tiebreaker when the magic bytes indicate a ZIP-based format.
    return _detect_type_from_bytes(header, filename=os.path.basename(file_path))


# ---------------------------------------------------------------------------
# 3. Private format loaders
# ---------------------------------------------------------------------------

def _load_docx(file_path: str) -> List[Dict[str, Any]]: #python-docx, one dict total
    try:
        import docx  # python-docx
    except ImportError as e:
        raise ImportError("python-docx is required to load DOCX files. Run: pip install python-docx") from e

    doc = docx.Document(file_path)
    # Join all non-empty paragraphs with newlines. DOCX has no concept of pages
    # in its paragraph model, so we treat the whole document as a single page —
    # the same approach used for TXT/MD. The chunker will split it further.
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    content = "\n".join(paragraphs).strip()
    if not content:
        return []  # empty document — nothing to index
    return [{
        "text": content,
        "source": os.path.basename(file_path),
        "page": 1,  # 1-based, consistent with PDF and TXT loaders
    }]


def _load_pdf(file_path: str) -> List[Dict[str, Any]]: #PyMuPDF, one dict per page
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("PyMuPDF is required to load PDF files. Run: pip install PyMuPDF") from e

    pages: List[Dict[str, Any]] = []
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            # Skip pages that have no extractable text (e.g. scanned images, blank pages).
            # Including them would add empty chunks to the index, wasting embedding space.
            if text:
                pages.append({
                    "text": text,
                    "source": os.path.basename(file_path),  # just the filename, not the full path
                    "page": page_num,  # 1-based to match what users see in a PDF viewer
                })
    return pages


def _load_text(file_path: str) -> List[Dict[str, Any]]: #plain read, one dict total
    # errors="replace" is intentional: it prevents a crash if the file contains
    # a handful of non-UTF-8 bytes (e.g. a Windows-1252 curly quote). The content
    # is still usable; we just substitute a replacement character for those bytes.
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read().strip()
    if not content:
        return []  # empty file — nothing to index
    # TXT and MD have no concept of pages in their file format, so we model
    # the whole document as a single page. The chunker will split it further.
    # We use page=1 (not 0) to stay consistent with the 1-based PDF page numbers.
    return [{
        "text": content,
        "source": os.path.basename(file_path),
        "page": 1,
    }]


# ---------------------------------------------------------------------------
# 4. Public loaders
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> List[Dict[str, Any]]: #file-on-disk entry point
    """Detect file type from content and dispatch to the correct loader."""
    # Detect by content, not extension — a file renamed from .pdf to .txt
    # would be routed correctly here, whereas extension-based routing would fail.
    file_type = _detect_type_from_path(file_path)

    if file_type == "pdf":
        return _load_pdf(file_path)
    elif file_type == "docx":
        return _load_docx(file_path)
    else:
        # Both .txt and .md reach the same loader — Markdown is plain text
        # as far as embedding is concerned; we don't need to parse its syntax.
        return _load_text(file_path)


# ---------------------------------------------------------------------------
# Convenience: load from bytes (Streamlit UploadedFile)
# ---------------------------------------------------------------------------

def load_from_bytes(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]: #Streamlit UploadedFile entry point
    """Detect file type from content, write to a temp file, load it, then clean up."""
    import tempfile

    # Detect type from the bytes directly, passing the original filename as a hint
    # so the ZIP-vs-DOCX tiebreaker in _detect_type_from_bytes works correctly.
    # We still rely primarily on magic bytes — the filename only resolves ambiguity
    # for ZIP-based formats (DOCX), not for PDF or plain-text detection.
    file_type = _detect_type_from_bytes(file_bytes, filename=filename)

    # Both PyMuPDF (fitz) and python-docx check the file extension when opening.
    # We give the temp file the correct extension based on detected type so the
    # underlying library doesn't reject or misparse it.
    if file_type == "pdf":
        suffix = ".pdf"
    elif file_type == "docx":
        suffix = ".docx"
    else:
        suffix = ".txt"

    # delete=False: on macOS/Linux the file can be opened again after the 'with'
    # block closes it, but we still need it to exist on disk for load_document.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        pages = load_document(tmp_path)
        # load_document sets source = the temp path (e.g. "/tmp/abc123.pdf").
        # We overwrite it with the original filename so the UI shows something
        # meaningful like "KB_v1.1.md" instead of a random temp path.
        for page in pages:
            page["source"] = filename
        return pages
    finally:
        # Use a finally block so the temp file is always deleted — even if
        # load_document raises an exception halfway through parsing.
        os.unlink(tmp_path)