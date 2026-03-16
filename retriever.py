"""
retriever.py — Advanced Retrieval
==================================

Purpose
-------
Bridge the user's raw question and the vector store.  The retriever's job is
to transform the question into the same embedding space as the stored chunks,
run a similarity search, and return a *rich result set* that includes not just
the matched text but also similarity scores and provenance metadata.

Why a dedicated retriever module?
----------------------------------
The retriever sits between the vector store and the generator.  Keeping it as
a separate class allows us to:

1.  **Enrich results** — add derived fields (rank, formatted citation) without
    polluting the vector store.
2.  **Log the full retrieval context** — the Streamlit "Under the Hood" panel
    reads from the retriever's last result set.
3.  **Future extensibility** — implement advanced strategies like Maximal
    Marginal Relevance (MMR) de-duplication, hypothetical document embeddings
    (HyDE), or multi-query fusion without changing the vector store.

Top-K retrieval explained
--------------------------
"Top-K" means: *retrieve the K most similar chunks to the query vector.*

*   K too small → the generator may lack enough context to answer accurately.
*   K too large → the generator's context window fills with marginally relevant
    or even noise chunks, diluting the signal and increasing hallucination risk.

A K of 3–7 is typically optimal for dense legal documents of moderate length.
The Streamlit sidebar exposes this as a configurable slider.

Similarity scores
-----------------
Each result carries a **cosine similarity score in [0, 1]** (converted from
ChromaDB's raw L2 distance by ``ChromaVectorStore.query``).

*   ≥ 0.85 — Highly relevant; almost certainly answers the question.
*   0.70–0.85 — Relevant; likely contains useful context.
*   0.50–0.70 — Marginally relevant; may contain peripheral context.
*   < 0.50 — Likely noise; consider raising the minimum score threshold.

Usage example
-------------
::

    from retriever import Retriever
    from vector_store import ChromaVectorStore
    from config import AppConfig

    cfg = AppConfig()
    store = ChromaVectorStore(cfg)
    retriever = Retriever(store, cfg)

    results = retriever.retrieve("What is the RRSP contribution limit for 2024?")
    for r in results:
        print(f"[{r.score:.3f}] {r.text[:100]}")
"""

from __future__ import annotations

import logging
from typing import Any

from config import AppConfig, FALLBACK_SCORE_THRESHOLD, SourceTier
from vector_store import BaseVectorStore, RetrievalResult

logger = logging.getLogger(__name__)


class Retriever:
    """
    Semantic retriever that wraps a :class:`~vector_store.BaseVectorStore`.

    The retriever is the entry point for the *R* in RAG.  It converts a
    natural-language question into an embedding, searches the vector store
    for semantically similar chunks, and returns them with rich metadata for
    both the generator and the educational UI.

    Parameters
    ----------
    vector_store:
        Any :class:`~vector_store.BaseVectorStore` implementation.
        Injecting the store (rather than constructing it internally) makes
        the retriever easy to unit-test with a mock store.
    config:
        Application configuration.  Used for the default ``top_k`` value.
    """

    def __init__(self, vector_store: BaseVectorStore, config: AppConfig) -> None:
        self._store = vector_store
        self._config = config

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant document chunks for *query*.

        Retrieval pipeline (step by step)
        ----------------------------------
        1.  **Query normalisation** — Strip leading/trailing whitespace and
            collapse internal whitespace runs.  A clean query produces a
            cleaner embedding.

        2.  **Embedding** — The (cleaned) query is embedded by the vector
            store's internal :class:`~vector_store.OpenAIEmbedder`.

        3.  **ANN search** — ChromaDB's HNSW index finds the *top_k* nearest
            stored vectors.  Optional *where* filter pre-scopes the search to
            a subset of the collection (e.g. a specific tax year).

        4.  **Score filtering** — Results below *min_score* are discarded.
            This optional threshold prevents low-quality matches from
            polluting the generator's context.

        5.  **Ranking** — Results are already ordered by similarity (highest
            first) by ChromaDB; we preserve this ordering.

        Parameters
        ----------
        query:
            The user's natural-language question.
        top_k:
            Number of chunks to retrieve.  Defaults to ``config.top_k``.
        where:
            Optional ChromaDB metadata filter (see
            :meth:`~vector_store.BaseVectorStore.query` for syntax).
        min_score:
            Minimum cosine similarity score (0–1).  Results below this
            threshold are excluded.  Default 0.0 (no filtering).

        Returns
        -------
        list[RetrievalResult]
            Retrieved chunks ordered from most to least similar.
            Empty list if the collection is empty or no matches exceed
            *min_score*.
        """
        # Normalise query
        clean_query = " ".join(query.strip().split())
        if not clean_query:
            logger.warning("Empty query passed to Retriever.retrieve().")
            return []

        k = top_k if top_k is not None else self._config.top_k

        # Guard against querying an empty collection
        chunk_count = self._store.count()
        if chunk_count == 0:
            logger.warning(
                "Vector store is empty. Please ingest documents first."
            )
            return []

        # Clamp k to the number of available chunks to avoid ChromaDB errors
        effective_k = min(k, chunk_count)

        logger.info(
            "Retrieving top-%d chunks for query: '%s' (filter=%s)",
            effective_k,
            clean_query[:80],
            where,
        )

        results = self._store.query(
            query_text=clean_query,
            top_k=effective_k,
            where=where,
        )

        # Apply optional minimum-score filter
        if min_score > 0.0:
            before = len(results)
            results = [r for r in results if r.score >= min_score]
            logger.debug(
                "Score filter (min=%.2f) removed %d low-quality results.",
                min_score,
                before - len(results),
            )

        logger.info(
            "Retrieval complete: %d results returned.", len(results)
        )
        return results

    def format_context(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a single context string for the generator.

        Each chunk is prefixed with its rank, similarity score, and source
        metadata so the LLM (and the user in the UI) can see provenance.

        Parameters
        ----------
        results:
            The list returned by :meth:`retrieve`.

        Returns
        -------
        str
            Multi-line context block ready to be injected into the prompt.
        """
        if not results:
            return "No relevant context found in the document collection."

        lines: list[str] = []
        for rank, r in enumerate(results, start=1):
            source = r.metadata.get("source", "Unknown source")
            tax_year = r.metadata.get("tax_year", "")
            doc_type = r.metadata.get("doc_type", "")

            # Build a compact citation string
            citation_parts = [f"Source: {source}"]
            if tax_year:
                citation_parts.append(f"Tax Year: {tax_year}")
            if doc_type:
                citation_parts.append(f"Type: {doc_type}")
            citation = " | ".join(citation_parts)

            lines.append(
                f"[Chunk {rank} | Similarity: {r.score:.4f} | {citation}]\n"
                f"{r.text}"
            )

        return "\n\n---\n\n".join(lines)

    def retrieve_with_fallback(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Two-stage retrieval: prefer OFFICIAL sources, fall back to COMMUNITY.

        Pipeline
        --------
        1.  Query the vector store filtered to ``SourceTier.OFFICIAL`` chunks.
        2.  If the best match has a similarity score below *threshold*
            (default :data:`~config.FALLBACK_SCORE_THRESHOLD`), issue a
            secondary query filtered to ``SourceTier.COMMUNITY`` chunks.
        3.  Return the winning result set together with a boolean flag
            indicating which tier was used.

        Parameters
        ----------
        query:
            The user's natural-language question.
        top_k:
            Number of chunks to retrieve per stage.  Defaults to
            ``config.top_k``.
        threshold:
            Minimum cosine-similarity score for an official result to be
            accepted.  Defaults to
            :data:`~config.FALLBACK_SCORE_THRESHOLD`.

        Returns
        -------
        dict
            ``{"results": list[RetrievalResult], "is_official": bool}``
        """
        k = top_k if top_k is not None else self._config.top_k
        score_threshold = (
            threshold if threshold is not None else FALLBACK_SCORE_THRESHOLD
        )

        # --- Step 1: OFFICIAL tier ---
        official_results = self.retrieve(
            query,
            top_k=k,
            where={"tier": SourceTier.OFFICIAL.value},
        )

        best_score = official_results[0].score if official_results else 0.0

        if best_score >= score_threshold:
            logger.info(
                "Official results accepted (best_score=%.4f >= %.4f).",
                best_score,
                score_threshold,
            )
            return {"results": official_results, "is_official": True}

        # --- Step 2: COMMUNITY fallback ---
        logger.info(
            "Official results too weak (best_score=%.4f < %.4f). "
            "Falling back to community sources.",
            best_score,
            score_threshold,
        )
        community_results = self.retrieve(
            query,
            top_k=k,
            where={"tier": SourceTier.COMMUNITY.value},
        )

        return {"results": community_results, "is_official": False}
