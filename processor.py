"""
processor.py — Document Ingestion & Chunking
============================================

Purpose
-------
Transform raw source documents (plain text files and PDFs) into a list of
text *chunks* that are small enough to embed efficiently and large enough
to contain useful context for retrieval.

Why chunking matters
--------------------
Language models have a finite *context window* (e.g. 128 k tokens for GPT-4o).
Even if we could fit an entire CRA folio into one prompt, we *shouldn't*:

1.  **Cost** — embedding and storing one giant vector per document means the
    retrieval step cannot distinguish between relevant and irrelevant passages.
2.  **Precision** — smaller chunks produce higher-quality similarity scores
    because the vector captures the meaning of a focused passage rather than
    a noisy average of hundreds of paragraphs.
3.  **Attribution** — chunk-level metadata lets us cite the exact page and
    section that supported an answer.

The sweet spot for dense legal/tax documents is typically 500–1 500 characters
with a 10–20 % overlap.

Architecture
------------
``DocumentProcessor`` depends on an abstract ``BaseChunker`` so that new
chunking strategies can be plugged in without modifying the processor itself
(Open/Closed Principle).

Two concrete chunkers are provided:

* ``FixedSizeChunker`` — Splits text at a fixed character boundary, then
  overlaps consecutive windows by ``overlap`` characters.  Fast and
  predictable, but may cut sentences mid-way.

* ``RecursiveCharacterChunker`` — Tries progressively shorter separators
  (paragraph → sentence → word → character) until each piece fits within
  ``chunk_size``.  **Preferred for legal documents** because it respects
  natural language boundaries, keeping complete sentences and numbered
  clauses together.

Usage example
-------------
::

    from processor import DocumentProcessor, RecursiveCharacterChunker
    from config import AppConfig

    cfg = AppConfig(chunk_size=1000, chunk_overlap=150)
    chunker = RecursiveCharacterChunker(cfg)
    processor = DocumentProcessor(chunker)

    chunks = processor.process_file("data/cra_t4_guide.pdf", {
        "doc_type": "T4 Guide",
        "tax_year": "2024",
        "source": "CRA",
    })
    print(f"Produced {len(chunks)} chunks")
"""

from __future__ import annotations

import abc
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# =============================================================================
# Public data contract
# =============================================================================


@dataclass
class DocumentChunk:
    """
    A single piece of text ready for embedding and storage.

    Attributes
    ----------
    text:
        The raw text content of this chunk.
    metadata:
        Arbitrary key-value pairs attached to this chunk.  These are stored
        alongside the vector in ChromaDB and allow downstream filtering
        (e.g. ``{"tax_year": "2024", "page": 3}``).
    chunk_index:
        Zero-based position of this chunk within its source document.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"DocumentChunk(index={self.chunk_index}, "
            f"chars={len(self.text)}, preview='{preview}...')"
        )


# =============================================================================
# Abstract chunker interface
# =============================================================================


class BaseChunker(abc.ABC):
    """
    Abstract base class for all chunking strategies.

    Concrete implementations must override :meth:`chunk`.

    Using an ABC here allows ``DocumentProcessor`` to accept *any* chunker
    without knowing its internals — this is the *Dependency Inversion*
    principle in practice.

    Parameters
    ----------
    chunk_size:
        Target maximum number of characters per chunk.
    overlap:
        Number of characters shared between the end of one chunk and the
        start of the next.  Overlap prevents important context from being
        lost at chunk boundaries.
    """

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abc.abstractmethod
    def chunk(self, text: str) -> list[str]:
        """
        Split *text* into a list of string chunks.

        Parameters
        ----------
        text:
            The full document text to split.

        Returns
        -------
        list[str]
            Ordered list of text chunks.  Order is preserved so that
            ``chunk_index`` values remain meaningful.
        """


# =============================================================================
# Concrete chunker: Fixed-Size Overlap
# =============================================================================


class FixedSizeChunker(BaseChunker):
    """
    Simple sliding-window chunker with fixed character sizes.

    Algorithm
    ---------
    1.  Start at position 0.
    2.  Take a slice of length ``chunk_size``.
    3.  Advance by ``chunk_size - overlap`` characters.
    4.  Repeat until the end of the document.

    Pros
    ----
    *   Extremely fast — O(n) in document length.
    *   Deterministic and easy to reason about.
    *   Works well when documents have uniform density (e.g. transcripts).

    Cons
    ----
    *   **Cuts sentences in half** — a chunk may start or end mid-sentence,
        which degrades embedding quality for structured prose.
    *   Poor fit for legal documents with numbered clauses (e.g.
        "1.(a)(ii) The amount described in subsection 20(1)...") because a
        clause boundary and a fixed-size boundary rarely coincide.

    When to use
    -----------
    Use ``FixedSizeChunker`` as a quick baseline or when your source material
    is already pre-segmented (e.g. CSV rows, structured JSON fields).
    """

    def chunk(self, text: str) -> list[str]:
        """
        Slide a fixed-size window over *text* with ``self.overlap`` characters
        of overlap between consecutive chunks.

        Parameters
        ----------
        text:
            Full document text.

        Returns
        -------
        list[str]
            List of text slices, each at most ``self.chunk_size`` characters.
        """
        chunks: list[str] = []
        step = self.chunk_size - self.overlap
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step

        logger.debug(
            "FixedSizeChunker produced %d chunks "
            "(chunk_size=%d, overlap=%d)",
            len(chunks),
            self.chunk_size,
            self.overlap,
        )
        return chunks


# =============================================================================
# Concrete chunker: Recursive Character Chunking
# =============================================================================


# Separator hierarchy for legal/tax documents (most preferred → least preferred)
_LEGAL_SEPARATORS: list[str] = [
    "\n\n\n",   # Major section break (e.g. between CRA folio sections)
    "\n\n",     # Paragraph break
    "\n",       # Line break (sub-paragraphs, numbered items)
    ". ",       # Sentence boundary
    "; ",       # Clause separator common in legislation
    ", ",       # Phrase separator
    " ",        # Word boundary (last resort before hard split)
    "",         # Hard character split (absolute last resort)
]


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursively splits text using a prioritised list of separators.

    Algorithm
    ---------
    Starting with the *most preferred* separator (large structural boundaries
    like ``\\n\\n\\n``), the chunker splits the text and checks whether each
    piece is already within ``chunk_size``.

    *   If a piece fits → keep it.
    *   If a piece is too large → recurse with the *next* separator in the
        hierarchy.
    *   If the piece still exceeds ``chunk_size`` after all separators are
        exhausted → fall back to a hard character split (same as
        ``FixedSizeChunker``).

    After all pieces are collected, adjacent pieces are *merged* if their
    combined length fits within ``chunk_size``, and the overlap between
    consecutive final chunks is reconstructed.

    Why this is better for legal/tax documents
    ------------------------------------------
    Government tax documents (CRA folios, T4 Guides, Income Tax Act excerpts)
    are structured as deeply nested prose with:

    *   Large top-level sections (separated by multiple blank lines).
    *   Numbered sub-sections and lettered items.
    *   Long, comma-heavy sentences that must stay intact for the meaning to
        be preserved (e.g. "Subject to subsection (3) and paragraph 60(b),
        there shall be included in computing the income of a taxpayer...").

    By preferring paragraph and sentence boundaries, ``RecursiveCharacterChunker``
    keeps these semantic units together.  The resulting vectors are more
    "meaningful" and retrieve more precisely because the embedding captures
    a coherent thought rather than an arbitrary character window.

    Parameters
    ----------
    chunk_size:
        Target maximum characters per chunk.
    overlap:
        Characters of overlap to reconstruct between final merged chunks.
    separators:
        Override the default separator hierarchy.  The list is ordered from
        *most preferred* (tried first) to *least preferred* (tried last).
    """

    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(chunk_size, overlap)
        self._separators = separators if separators is not None else _LEGAL_SEPARATORS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[str]:
        """
        Recursively chunk *text* using the separator hierarchy.

        Parameters
        ----------
        text:
            Full document text.

        Returns
        -------
        list[str]
            Ordered list of text chunks respecting natural boundaries.
        """
        raw_pieces = self._recursive_split(text, self._separators)
        merged = self._merge_with_overlap(raw_pieces)

        logger.debug(
            "RecursiveCharacterChunker produced %d chunks "
            "(chunk_size=%d, overlap=%d)",
            len(merged),
            self.chunk_size,
            self.overlap,
        )
        return merged

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Split *text* with the first separator that produces pieces within
        ``chunk_size``.  Recurse on oversized pieces with the remaining
        separators.

        Parameters
        ----------
        text:
            Text to split.
        separators:
            Remaining separator hierarchy (shrinks with each recursion level).

        Returns
        -------
        list[str]
            Flat list of text pieces, each at most ``chunk_size`` characters.
        """
        if not text.strip():
            return []

        # Base case: no separators left — hard character split
        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        rest = separators[1:]

        # Split on the current separator
        if sep:
            parts = text.split(sep)
        else:
            parts = list(text)

        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # Too large — recurse with the next separator
                result.extend(self._recursive_split(part, rest))

        return result

    def _hard_split(self, text: str) -> list[str]:
        """
        Fallback: split *text* into fixed-size pieces when no separator works.

        Parameters
        ----------
        text:
            Text to split.

        Returns
        -------
        list[str]
            Fixed-size character slices.
        """
        return [
            text[i: i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """
        Greedily merge consecutive small pieces into chunks up to
        ``chunk_size``, then reconstruct the specified overlap between
        adjacent final chunks.

        This two-pass approach ensures:

        1.  Chunks are as large as possible (good embedding density).
        2.  Adjacent chunks share ``self.overlap`` characters of context
            so that sentences spanning a boundary are covered by at least
            one chunk.

        Parameters
        ----------
        pieces:
            Flat list of small text pieces from ``_recursive_split``.

        Returns
        -------
        list[str]
            Merged chunks with overlap.
        """
        if not pieces:
            return []

        merged: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for piece in pieces:
            piece_length = len(piece)
            # +1 for the space/newline separator we'll add when joining
            if current_parts and (current_length + piece_length + 1) > self.chunk_size:
                # Flush current accumulator
                merged.append(" ".join(current_parts))
                # Seed the next chunk with the overlap tail of the flushed chunk
                overlap_text = merged[-1][-self.overlap:] if self.overlap else ""
                current_parts = [overlap_text, piece] if overlap_text else [piece]
                current_length = len(overlap_text) + piece_length
            else:
                current_parts.append(piece)
                current_length += piece_length + (1 if len(current_parts) > 1 else 0)

        # Don't forget the last accumulator
        if current_parts:
            merged.append(" ".join(current_parts))

        return [c for c in merged if c.strip()]


# =============================================================================
# Document loader helpers
# =============================================================================


def _load_text_file(path: Path) -> str:
    """
    Read a plain-text file and return its content as a string.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.txt`` file.

    Returns
    -------
    str
        Full file content decoded as UTF-8, with Windows-style line endings
        normalised to Unix-style.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    UnicodeDecodeError
        If the file is not valid UTF-8.
    """
    logger.info("Loading text file: %s", path)
    content = path.read_text(encoding="utf-8")
    # Normalise CRLF → LF for consistent splitting behaviour
    return content.replace("\r\n", "\n")


def _load_pdf_file(path: Path) -> str:
    """
    Extract all text from a PDF file using ``pypdf``.

    Extraction strategy
    -------------------
    ``pypdf`` uses a layout-aware extraction mode that preserves paragraph
    structure better than a raw byte stream.  For government PDFs (which are
    usually text-based rather than scanned images), this produces clean prose
    with natural paragraph breaks.

    Note on scanned PDFs
    --------------------
    If the CRA document is a *scanned image* PDF (no embedded text layer),
    ``pypdf`` will return empty strings.  In that case, an OCR step (e.g.
    ``pytesseract``) would be required before chunking — a future extension
    point in this playground.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.pdf`` file.

    Returns
    -------
    str
        Concatenated text from all pages, separated by ``\\n\\n``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ImportError
        If ``pypdf`` is not installed.
    """
    try:
        import pypdf  # noqa: PLC0415 — lazy import to keep startup fast
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to load PDF files. "
            "Install it with: pip install pypdf"
        ) from exc

    logger.info("Loading PDF file: %s (%d pages)", path, 0)
    pages: list[str] = []

    with pypdf.PdfReader(str(path)) as reader:
        logger.info("Loading PDF file: %s (%d pages)", path, len(reader.pages))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # Remove excessive whitespace while preserving paragraph structure
            text = re.sub(r" {2,}", " ", text)
            if text.strip():
                pages.append(f"[Page {page_num}]\n{text.strip()}")

    return "\n\n".join(pages)


# =============================================================================
# Document Processor (orchestrator)
# =============================================================================


class DocumentProcessor:
    """
    Orchestrates loading raw documents and converting them into
    :class:`DocumentChunk` objects ready for embedding.

    Design
    ------
    ``DocumentProcessor`` is intentionally *thin*: it delegates file-reading
    to format-specific helpers and text-splitting to the injected
    ``BaseChunker`` implementation.  This separation means:

    *   Adding a new file format (e.g. ``.docx``) only requires adding a new
        loader function and registering it in :meth:`_load_raw_text`.
    *   Swapping the chunking strategy requires only passing a different
        ``BaseChunker`` to the constructor — the processor code is unchanged.

    Parameters
    ----------
    chunker:
        A :class:`BaseChunker` instance that defines *how* the document is
        split.  Inject ``RecursiveCharacterChunker`` for legal documents.
    """

    #: Mapping from file extension to loader callable.
    _LOADERS: dict[str, Any] = {
        ".txt": _load_text_file,
        ".md": _load_text_file,
        ".pdf": _load_pdf_file,
    }

    def __init__(self, chunker: BaseChunker) -> None:
        self._chunker = chunker

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk an already-loaded string and return :class:`DocumentChunk` objects.

        Use this method when you have text content that was obtained from a
        source other than a local file (e.g. a database, an API, or a
        Streamlit text area).

        Parameters
        ----------
        text:
            Full document text.
        metadata:
            Arbitrary key-value pairs to attach to every chunk produced from
            this text (e.g. ``{"source": "manual_entry", "tax_year": "2024"}``).

        Returns
        -------
        list[DocumentChunk]
            Ordered list of chunks with injected metadata and sequential indices.
        """
        meta = metadata or {}
        raw_chunks = self._chunker.chunk(text)
        return [
            DocumentChunk(text=c, metadata=dict(meta), chunk_index=i)
            for i, c in enumerate(raw_chunks)
        ]

    def process_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Load a file from disk, chunk it, and return :class:`DocumentChunk` objects.

        Supported formats: ``.txt``, ``.md``, ``.pdf``

        Parameters
        ----------
        file_path:
            Path to the source document.
        metadata:
            Extra key-value pairs merged into each chunk's metadata.  The
            ``source`` key is automatically set to the file name if not
            already present.

        Returns
        -------
        list[DocumentChunk]
            Ordered list of chunks.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If the file extension is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        loader = self._LOADERS.get(suffix)
        if loader is None:
            supported = ", ".join(self._LOADERS)
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported formats: {supported}"
            )

        raw_text = loader(path)

        # Enrich metadata with file-level information
        meta: dict[str, Any] = {"source": path.name}
        if metadata:
            meta.update(metadata)

        logger.info(
            "Processing '%s' — %d characters → chunking with %s",
            path.name,
            len(raw_text),
            type(self._chunker).__name__,
        )
        return self.process_text(raw_text, meta)


# =============================================================================
# URL / Web Processor
# =============================================================================


class WebProcessor:
    """
    Fetch a URL, extract clean text with BeautifulSoup, and chunk it.

    Each produced :class:`DocumentChunk` carries metadata including the
    source ``url`` and the :class:`~config.SourceTier` value determined
    from the URL's domain via :data:`~config.DOMAIN_TIER_MAP`.

    Parameters
    ----------
    chunker:
        A :class:`BaseChunker` instance used to split the extracted text.
    """

    def __init__(self, chunker: BaseChunker) -> None:
        self._chunker = chunker
        self._processor = DocumentProcessor(chunker)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_tier(url: str) -> "SourceTier":
        """
        Return the :class:`~config.SourceTier` for *url* based on its domain.

        The lookup checks whether any key in
        :data:`~config.DOMAIN_TIER_MAP` appears in the URL's hostname.
        If no match is found the tier defaults to ``COMMUNITY``.
        """
        from config import DOMAIN_TIER_MAP, SourceTier  # noqa: PLC0415

        hostname = urlparse(url).hostname or ""
        for domain, tier in DOMAIN_TIER_MAP.items():
            if domain in hostname:
                return tier
        return SourceTier.COMMUNITY

    @staticmethod
    def _extract_text(url: str) -> str:
        """
        Fetch *url* and return its visible text content.

        Uses ``requests`` for the HTTP call and ``BeautifulSoup`` for HTML
        parsing.  Script and style elements are removed before extraction.

        Raises
        ------
        ImportError
            If ``requests`` or ``beautifulsoup4`` are not installed.
        requests.HTTPError
            If the server returns a non-2xx status code.
        """
        try:
            import requests  # noqa: PLC0415
            from bs4 import BeautifulSoup  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "requests and beautifulsoup4 are required for URL processing. "
                "Install them with: pip install requests beautifulsoup4"
            ) from exc

        logger.info("Fetching URL: %s", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_url(
        self,
        url: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Fetch, extract, chunk, and tag content from *url*.

        Every chunk produced carries at least ``{"url": url, "tier": …}``
        in its metadata so that the retriever can filter by source tier.

        Parameters
        ----------
        url:
            The web page to ingest.
        metadata:
            Extra key-value pairs merged into each chunk's metadata.

        Returns
        -------
        list[DocumentChunk]
            Ordered list of chunks with injected URL and tier metadata.
        """
        tier = self._determine_tier(url)
        raw_text = self._extract_text(url)

        meta: dict[str, Any] = {
            "url": url,
            "tier": tier.value,
            "source": url,
        }
        if metadata:
            meta.update(metadata)

        logger.info(
            "Processing URL '%s' — %d characters, tier=%s",
            url,
            len(raw_text),
            tier.value,
        )
        return self._processor.process_text(raw_text, meta)
