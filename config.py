"""
config.py — Shared Configuration & Constants
=============================================

Purpose
-------
Centralise all tuneable parameters, environment variables, and application-wide
constants in one place.  Every other module imports from here rather than
hard-coding values, which makes the playground easy to reconfigure without
touching business logic.

Design rationale
----------------
*   A single ``AppConfig`` dataclass acts as a dependency-injection vehicle:
    callers create one instance (potentially with overridden values from the
    Streamlit sidebar) and pass it down the call stack.
*   Environment variables are read once at import time via ``python-dotenv``.
    The OpenAI key is **never** stored in application state or logged.
*   Default values are chosen to work well on commodity hardware (small chunk
    size, moderate Top-K) while still demonstrating the concepts clearly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env file if it exists (keeps secrets out of source control)
load_dotenv()


# ---------------------------------------------------------------------------
# OpenAI model identifiers
# ---------------------------------------------------------------------------

#: Embedding model used to convert text chunks into vectors.
#: "text-embedding-3-small" offers a great accuracy/cost trade-off for RAG.
EMBEDDING_MODEL: str = "text-embedding-3-small"

#: Chat completion model used for the final answer generation step.
#: "gpt-4o-mini" is fast and cost-effective; swap to "gpt-4o" for higher quality.
CHAT_MODEL: str = "gpt-4o-mini"

#: Dimension of vectors produced by ``EMBEDDING_MODEL``.
#: Required when creating a ChromaDB collection with a custom embedding function.
EMBEDDING_DIMENSION: int = 1536


# ---------------------------------------------------------------------------
# ChromaDB persistence settings
# ---------------------------------------------------------------------------

#: Directory where ChromaDB will persist its SQLite + binary index files.
#: Using a subdirectory keeps the repo root tidy.
CHROMA_PERSIST_DIR: str = "./chroma_db"

#: Name of the ChromaDB collection that stores the Canadian Tax document chunks.
CHROMA_COLLECTION_NAME: str = "canadian_tax_docs"


# ---------------------------------------------------------------------------
# Chunking defaults
# ---------------------------------------------------------------------------

#: Default number of tokens/characters per chunk.
#: 1 000 characters ≈ 200-250 tokens, which fits comfortably within context windows
#: while still containing enough context for meaningful retrieval.
DEFAULT_CHUNK_SIZE: int = 1000

#: Default overlap between consecutive chunks (characters).
#: Overlap prevents important sentences that straddle a chunk boundary from
#: being split across two chunks, which would degrade retrieval quality.
DEFAULT_CHUNK_OVERLAP: int = 150


# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

#: Default number of chunks to retrieve for each user query (Top-K).
DEFAULT_TOP_K: int = 5


# ---------------------------------------------------------------------------
# Application-wide configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    """
    Immutable (by convention) configuration object passed throughout the app.

    Parameters
    ----------
    openai_api_key:
        OpenAI secret key.  Defaults to the ``OPENAI_API_KEY`` environment
        variable.  Never log or display this value.
    embedding_model:
        OpenAI embedding model identifier.
    chat_model:
        OpenAI chat completion model identifier.
    chroma_persist_dir:
        File-system path for ChromaDB's persistent storage.
    collection_name:
        Name of the ChromaDB collection to use.
    chunk_size:
        Target character count per document chunk.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.
    top_k:
        Number of similar chunks to retrieve per query.
    """

    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "")
    )
    embedding_model: str = EMBEDDING_MODEL
    chat_model: str = CHAT_MODEL
    chroma_persist_dir: str = CHROMA_PERSIST_DIR
    collection_name: str = CHROMA_COLLECTION_NAME
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K

    def is_configured(self) -> bool:
        """Return ``True`` if a non-empty OpenAI API key is present."""
        return bool(self.openai_api_key)
