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
from enum import Enum
from typing import TypedDict

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
#: Kept for backward compatibility; personas override this per domain.
CHROMA_COLLECTION_NAME: str = "canadian_tax_docs"


# ---------------------------------------------------------------------------
# Source-tier classification for URL ingestion
# ---------------------------------------------------------------------------


class SourceTier(Enum):
    """
    Classify ingested content by its authoritativeness.

    ``OFFICIAL`` sources (e.g. canada.ca) are treated as primary ground truth
    during retrieval.  ``COMMUNITY`` sources (e.g. Reddit, canadavisa.com) are
    used only as a fallback when no strong official match exists.
    """

    OFFICIAL = "official"
    COMMUNITY = "community"


#: Maps domain substrings to their :class:`SourceTier`.  The retriever and
#: web processor use this mapping to tag every ingested chunk with the
#: correct tier.  Add new domains here to extend coverage.
DOMAIN_TIER_MAP: dict[str, SourceTier] = {
    "canada.ca": SourceTier.OFFICIAL,
    "reddit.com": SourceTier.COMMUNITY,
    "canadavisa.com": SourceTier.COMMUNITY,
}


# ---------------------------------------------------------------------------
# Retrieval fallback threshold
# ---------------------------------------------------------------------------

#: Minimum cosine-similarity score for an OFFICIAL-tier result to be
#: considered "good enough".  If no official chunk meets this threshold the
#: retriever falls back to COMMUNITY-tier results.
FALLBACK_SCORE_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# Persona definitions — one entry per supported domain
# ---------------------------------------------------------------------------


class PersonaConfig(TypedDict):
    """Typed structure for a single persona entry in :data:`PERSONAS`."""

    collection_name: str
    ui_title: str
    system_prompt: str


#: Each persona bundles a ChromaDB collection name, a UI title, and a
#: domain-specific system prompt template.  The ``{context}`` placeholder
#: in every ``system_prompt`` is replaced at runtime with the retrieved
#: document chunks.

PERSONAS: dict[str, PersonaConfig] = {
    "Tax Assistant": {
        "collection_name": "tax_kb",
        "ui_title": "🍁 Canadian Tax Assistant",
        "system_prompt": (
            "You are a highly specialised Canadian Tax Assistant with deep "
            "expertise in the Canada Revenue Agency (CRA) tax regulations, "
            "the Income Tax Act (Canada), and related CRA publications.\n\n"
            "YOUR SOLE SOURCE OF TRUTH\n"
            "==========================\n"
            "You must answer ONLY using the context excerpts provided below. "
            "These excerpts have been retrieved from official CRA documents "
            "and are the only information you are permitted to use when "
            "formulating your answer.\n\n"
            "RETRIEVED CRA DOCUMENT CONTEXT\n"
            "================================\n"
            "{context}\n\n"
            "STRICT RULES — READ CAREFULLY\n"
            "==============================\n"
            "1. ONLY answer based on the context above. Do NOT use any "
            "outside knowledge.\n"
            "2. NEVER reference, mention, or confuse Canadian tax rules with "
            "US tax rules (IRS, 401(k), Roth IRA, W-2, etc.). These are "
            "completely different systems.\n"
            "3. If the answer to the question is NOT present in the provided "
            "context, you MUST respond with exactly: \"I cannot find "
            "information about that in the provided CRA documents. Please "
            "consult a qualified Canadian tax professional or visit the CRA "
            "website directly.\"\n"
            "4. When you cite a fact, always mention its source document if "
            "it appears in the context metadata.\n"
            "5. Do NOT speculate, extrapolate, or fill in gaps with "
            "assumptions.\n"
            "6. Be concise and precise. Tax law is nuanced — do not "
            "over-simplify."
        ),
    },
    "Immigration Assistant": {
        "collection_name": "immigration_kb",
        "ui_title": "🛂 Canadian Immigration Assistant (IMM Forms)",
        "system_prompt": (
            "You are a knowledgeable Canadian Immigration Assistant "
            "specialising in Immigration, Refugees and Citizenship Canada "
            "(IRCC) processes and forms such as IMM 5257 (Application to "
            "Visit Canada) and IMM 5707 (Family Information).\n\n"
            "YOUR SOLE SOURCE OF TRUTH\n"
            "==========================\n"
            "You must answer ONLY using the context excerpts provided below. "
            "These excerpts have been retrieved from official IRCC / "
            "canada.ca documents and are the only information you are "
            "permitted to use when formulating your answer.\n\n"
            "RETRIEVED IRCC DOCUMENT CONTEXT\n"
            "================================\n"
            "{context}\n\n"
            "STRICT RULES — READ CAREFULLY\n"
            "==============================\n"
            "1. ONLY answer based on the context above. Do NOT use any "
            "outside knowledge.\n"
            "2. Act as a step-by-step guide to help users understand and "
            "fill out Canadian visa / immigration forms (e.g. IMM 5257, "
            "IMM 5707) based strictly on provided IRCC / canada.ca "
            "context.\n"
            "3. If the answer to the question is NOT present in the provided "
            "context, you MUST respond with exactly: \"I cannot find "
            "information about that in the provided IRCC documents. Please "
            "consult a licensed immigration consultant (RCIC) or visit "
            "canada.ca directly.\"\n"
            "4. When you cite a fact, always mention its source document if "
            "it appears in the context metadata.\n"
            "5. Do NOT speculate, extrapolate, or fill in gaps with "
            "assumptions.\n"
            "6. Be concise and precise. Immigration law is nuanced — do not "
            "over-simplify."
        ),
    },
}

#: The persona selected by default when the application first loads.
DEFAULT_PERSONA: str = "Tax Assistant"


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
