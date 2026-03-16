"""
vector_store.py — Embedding & Vector Storage
============================================

Purpose
-------
Convert :class:`~processor.DocumentChunk` objects into numerical vectors
(embeddings) and persist them in a vector database so they can be searched
by semantic similarity at query time.

Core concepts
-------------
**Embeddings**
    An embedding is a dense, fixed-length numerical vector that captures the
    *meaning* of a piece of text.  Two semantically similar texts produce
    vectors that are geometrically close in the high-dimensional embedding
    space.  For example, "RRSP contribution limit" and "Registered Retirement
    Savings Plan deduction ceiling" will be close together even though they
    share no words.

**Vector Database**
    A specialised database optimised for *approximate nearest-neighbour* (ANN)
    search.  Given a query vector, it efficiently returns the *k* stored
    vectors that are most similar.  ChromaDB stores both the vectors *and* the
    original text + metadata, so we can return the matching document chunks
    directly.

**Metadata Filtering**
    Before performing ANN search, ChromaDB can apply an exact-match *pre-filter*
    on metadata fields.  This lets us scope a query to a specific tax year,
    document type, or CRA folio section without scanning the whole collection —
    essential for compliance applications where answers must come from a
    specific regulatory version.

Architecture
------------
An abstract ``BaseVectorStore`` defines the interface so that alternative
backends (Pinecone, Weaviate, FAISS, etc.) can be plugged in without changing
any downstream code.  ``ChromaVectorStore`` is the concrete implementation
backed by a local, embedded ChromaDB instance.

Usage example
-------------
::

    from vector_store import ChromaVectorStore
    from config import AppConfig

    cfg = AppConfig()
    store = ChromaVectorStore(cfg)

    # Add chunks produced by DocumentProcessor
    store.add_chunks(chunks, batch_size=100)

    # Query with optional metadata filter
    results = store.query(
        query_text="RRSP contribution limit 2024",
        top_k=5,
        where={"tax_year": "2024"},
    )
    for r in results:
        print(r.score, r.text[:80])
"""

from __future__ import annotations

import abc
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import AppConfig
from processor import DocumentChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Public data contract
# =============================================================================


@dataclass
class RetrievalResult:
    """
    A single search result returned by :meth:`BaseVectorStore.query`.

    Attributes
    ----------
    text:
        The original chunk text.
    metadata:
        Key-value pairs stored alongside the chunk when it was ingested
        (e.g. ``{"source": "t4_guide.pdf", "tax_year": "2024", "page": 3}``).
    score:
        Similarity score in the range ``[0, 1]`` where **1 is most similar**.
        ChromaDB natively returns L2 *distance* (lower = more similar);
        this class converts it to a cosine-like similarity for readability.
    chunk_id:
        The unique identifier assigned to this chunk in the vector store.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    chunk_id: str = ""

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"RetrievalResult(score={self.score:.4f}, "
            f"preview='{preview}...')"
        )


# =============================================================================
# Abstract interface
# =============================================================================


class BaseVectorStore(abc.ABC):
    """
    Abstract base class for all vector store backends.

    Concrete implementations must provide :meth:`add_chunks`,
    :meth:`query`, :meth:`delete_collection`, and :meth:`count`.

    Coding to this interface rather than to a specific backend means the
    rest of the application is completely decoupled from ChromaDB.  Swapping
    to Pinecone or FAISS requires only a new subclass — no other files change.
    """

    @abc.abstractmethod
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 100,
    ) -> None:
        """
        Embed and persist a list of :class:`~processor.DocumentChunk` objects.

        Parameters
        ----------
        chunks:
            Chunks produced by :class:`~processor.DocumentProcessor`.
        batch_size:
            Number of chunks to embed and upsert per API call.  Smaller
            batches reduce memory pressure; larger batches reduce API
            round-trips.  The OpenAI embedding API accepts up to 2 048
            inputs per request.
        """

    @abc.abstractmethod
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Return the *top_k* most similar chunks to *query_text*.

        Parameters
        ----------
        query_text:
            The user's question or search string.
        top_k:
            Maximum number of results to return.
        where:
            Optional ChromaDB-compatible metadata filter expression.
            See https://docs.trychroma.com/usage-guide#filtering-by-metadata
            Examples::

                # Single field equality
                where={"tax_year": "2024"}

                # Multiple fields (implicit AND)
                where={"tax_year": "2024", "doc_type": "T4 Guide"}

                # Explicit operator
                where={"tax_year": {"$in": ["2023", "2024"]}}

        Returns
        -------
        list[RetrievalResult]
            Ordered from most similar to least similar.
        """

    @abc.abstractmethod
    def delete_collection(self) -> None:
        """
        Permanently delete all documents in the collection.

        Use with care — this is a destructive operation that cannot be undone
        without re-ingesting all documents.
        """

    @abc.abstractmethod
    def count(self) -> int:
        """
        Return the total number of chunks currently stored in the collection.

        Returns
        -------
        int
            Chunk count (0 if the collection is empty or does not yet exist).
        """


# =============================================================================
# OpenAI Embedding helper
# =============================================================================


class OpenAIEmbedder:
    """
    Thin wrapper around the OpenAI embeddings API.

    Separating the embedding logic from the vector store makes it easy to
    swap in a local embedding model (e.g. ``sentence-transformers``) later:
    just implement the same ``embed`` interface and inject it into
    ``ChromaVectorStore``.

    Parameters
    ----------
    api_key:
        OpenAI secret key.
    model:
        Embedding model name (default: ``"text-embedding-3-small"``).
    """

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts and return their vector representations.

        The OpenAI API normalises the returned vectors to unit length
        (L2-norm = 1), which means that L2 distance and cosine distance are
        equivalent, and a dot product is equivalent to cosine similarity.

        Parameters
        ----------
        texts:
            List of strings to embed.  Each string should be at most
            ~8 191 tokens (the model's input limit).

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, in the same order.
        """
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        # The API guarantees results are returned in the same order as inputs
        return [item.embedding for item in response.data]


# =============================================================================
# ChromaDB implementation
# =============================================================================


class ChromaVectorStore(BaseVectorStore):
    """
    Concrete vector store backed by a *local* ChromaDB instance.

    ChromaDB overview
    -----------------
    ChromaDB is an open-source embedding database that runs entirely in-process
    (no separate server required) and persists data to disk as SQLite + binary
    index files.  This makes it ideal for educational projects and local
    development — no infrastructure setup needed.

    Distance metric
    ---------------
    ChromaDB uses **L2 (Euclidean) distance** by default:

    .. math::

        d(u, v) = \\sqrt{\\sum_i (u_i - v_i)^2}

    Lower distance = higher similarity.

    Because OpenAI embeddings are unit-normalised, we can convert L2 distance
    to a [0, 1] similarity score using:

    .. math::

        \\text{similarity} = 1 - \\frac{d^2}{2}

    This is an exact identity for unit vectors:
    ``||u - v||^2 = 2 - 2·cos(u,v)`` → ``cos(u,v) = 1 - d²/2``

    Metadata storage
    ----------------
    ChromaDB stores arbitrary string/int/float metadata alongside each vector.
    Filtering is applied *before* the ANN search, which is much faster than
    post-filtering (especially for large collections).

    Parameters
    ----------
    config:
        Application configuration containing the API key, model names,
        persist directory, and collection name.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._embedder = OpenAIEmbedder(
            api_key=config.openai_api_key,
            model=config.embedding_model,
        )
        self._client = self._make_client()
        self._collection = self._get_or_create_collection()

    # ------------------------------------------------------------------
    # BaseVectorStore implementation
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 100,
    ) -> None:
        """
        Embed and upsert chunks into the ChromaDB collection.

        Batching
        --------
        The OpenAI embedding API has a per-request input limit, and embedding
        thousands of chunks in a single call would consume too much memory.
        We process ``batch_size`` chunks at a time to keep peak memory usage
        bounded.

        Upsert semantics
        ----------------
        ChromaDB's ``upsert`` creates a new entry if the ID doesn't exist, or
        updates the entry if it does.  We derive IDs deterministically from
        ``chunk_index`` + ``source``, so re-ingesting the same file is idempotent.

        Parameters
        ----------
        chunks:
            Chunks to store.
        batch_size:
            Chunks per embedding API call and ChromaDB upsert.
        """
        if not chunks:
            logger.warning("add_chunks called with an empty list — nothing to do.")
            return

        logger.info("Ingesting %d chunks in batches of %d ...", len(chunks), batch_size)

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start: batch_start + batch_size]

            texts = [c.text for c in batch]
            metadatas = [self._sanitise_metadata(c.metadata) for c in batch]
            ids = [self._make_chunk_id(c) for c in batch]

            # --- Embedding step ---
            logger.debug("Embedding batch %d–%d ...", batch_start, batch_start + len(batch))
            embeddings = self._embedder.embed(texts)

            # --- Storage step ---
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(
                "Upserted %d chunks (total so far: %d)",
                len(batch),
                batch_start + len(batch),
            )

        logger.info("Ingestion complete. Collection now contains %d chunks.", self.count())

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Embed *query_text* and retrieve the *top_k* nearest neighbours.

        How retrieval works (step by step)
        -----------------------------------
        1.  **Query embedding** — The user's question is embedded using the
            same model that was used during ingestion.  This is crucial: the
            query and document vectors must live in the same embedding space
            for distance comparisons to be meaningful.

        2.  **ANN search** — ChromaDB performs an approximate nearest-neighbour
            search over all stored vectors using its HNSW index.  The
            ``top_k`` vectors geometrically closest to the query vector are
            returned, along with their L2 distances.

        3.  **Score conversion** — Raw L2 distances are converted to
            cosine-similarity scores in ``[0, 1]`` (1 = identical) using
            ``similarity = 1 - d² / 2`` (valid for unit-norm vectors).

        4.  **Metadata filter** (optional) — If *where* is provided, ChromaDB
            applies it as a pre-filter, restricting the search to chunks that
            match the filter expression.

        Parameters
        ----------
        query_text:
            The user's question or search string.
        top_k:
            Maximum number of results.
        where:
            ChromaDB metadata filter (see class docstring for examples).

        Returns
        -------
        list[RetrievalResult]
            Ordered from most similar (highest score) to least similar.
        """
        logger.debug("Embedding query: '%s'", query_text[:80])

        # Guard: if the collection is empty, return immediately
        count = self.count()
        if count == 0:
            logger.warning("query() called on an empty collection — returning [].")
            return []

        query_embedding = self._embedder.embed([query_text])[0]

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        results: list[RetrievalResult] = []
        # ChromaDB returns nested lists (one per query); we sent a single query
        for doc, meta, dist, chunk_id in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
            raw["ids"][0],
        ):
            # Convert L2 distance to cosine similarity (valid for unit vectors)
            # d² = 2 - 2·cos → cos = 1 - d²/2
            similarity = max(0.0, 1.0 - (dist ** 2) / 2.0)
            results.append(
                RetrievalResult(
                    text=doc,
                    metadata=meta or {},
                    score=round(similarity, 6),
                    chunk_id=chunk_id,
                )
            )

        logger.debug("Query returned %d results.", len(results))
        return results

    def delete_collection(self) -> None:
        """
        Delete and recreate the ChromaDB collection, removing all chunks.

        This is useful in the Streamlit UI when the user wants to re-ingest
        documents with different settings (e.g. a new chunk size).
        """
        logger.warning(
            "Deleting ChromaDB collection '%s'.", self._config.collection_name
        )
        self._client.delete_collection(self._config.collection_name)
        self._collection = self._get_or_create_collection()

    def count(self) -> int:
        """
        Return the number of chunks stored in the collection.

        Returns
        -------
        int
            Current chunk count.
        """
        return self._collection.count()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_client(self):
        """
        Instantiate a persistent ChromaDB client.

        ``chromadb.PersistentClient`` stores all data in the directory
        specified by ``config.chroma_persist_dir``.  The data survives
        process restarts, so re-ingestion is only needed when the source
        documents change or the configuration changes.
        """
        try:
            import chromadb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "chromadb package is required. Install with: pip install chromadb"
            ) from exc

        return chromadb.PersistentClient(path=self._config.chroma_persist_dir)

    def _get_or_create_collection(self):
        """
        Retrieve an existing ChromaDB collection or create a new one.

        We use the **cosine** distance function here.  Although OpenAI vectors
        are already unit-normalised (making L2 and cosine equivalent), being
        explicit avoids surprises if a different embedding model is swapped in.
        """
        try:
            import chromadb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "chromadb package is required. Install with: pip install chromadb"
            ) from exc

        return self._client.get_or_create_collection(
            name=self._config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _make_chunk_id(chunk: DocumentChunk) -> str:
        """
        Generate a deterministic, unique ID for a chunk.

        Deriving the ID from the chunk's source and index (rather than using
        a random UUID) makes upserts idempotent — ingesting the same file
        twice simply overwrites the existing entries rather than creating
        duplicates.

        Parameters
        ----------
        chunk:
            The chunk for which to generate an ID.

        Returns
        -------
        str
            A unique string ID.
        """
        source = chunk.metadata.get("source", "unknown")
        return f"{source}::chunk_{chunk.chunk_index}"

    @staticmethod
    def _sanitise_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure all metadata values are ChromaDB-compatible types.

        ChromaDB only accepts ``str``, ``int``, ``float``, and ``bool`` as
        metadata values.  Any other type (lists, nested dicts, ``None``) is
        converted to its string representation to avoid a runtime error.

        Parameters
        ----------
        metadata:
            Raw metadata dict from a :class:`~processor.DocumentChunk`.

        Returns
        -------
        dict[str, Any]
            Sanitised metadata where all values are ChromaDB-safe.
        """
        safe: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif v is None:
                safe[k] = ""
            else:
                safe[k] = str(v)
        return safe
