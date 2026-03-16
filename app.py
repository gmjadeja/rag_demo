"""
app.py — Universal RAG Playground: Educational Streamlit UI
============================================================

Purpose
-------
This is the main entry point for the Universal RAG Playground.  It exposes
every step of the RAG pipeline visually so developers can observe *exactly*
what happens between a user question and the final answer.

Key educational features
------------------------
*   **Sidebar configurator** — Adjust chunk size, overlap, and Top-K
    retrieval without editing code.
*   **Document ingestion panel** — Upload PDFs/text files or use the bundled
    CRA sample document.
*   **Chat interface** — Standard conversational UI backed by the full RAG
    pipeline.
*   **"Under the Hood" expander** — For every response, the UI reveals:
    - The exact CRA document chunks that were retrieved.
    - The cosine similarity score for each chunk.
    - The complete prompt sent to the LLM (system + user messages).

How to run
----------
::

    streamlit run app.py

Environment variables
---------------------
Set ``OPENAI_API_KEY`` in a ``.env`` file or the Streamlit secrets manager.
The app will prompt for the key in the sidebar if it is not found.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Logging — visible in the terminal running Streamlit
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local module imports (all in the same directory)
# ---------------------------------------------------------------------------
from config import AppConfig, DEFAULT_PERSONA, PERSONAS
from processor import DocumentProcessor, RecursiveCharacterChunker, FixedSizeChunker
from vector_store import ChromaVectorStore
from retriever import Retriever
from generator import RAGGenerator

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal RAG Playground",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Session-state helpers
# =============================================================================

def _init_session_state() -> None:
    """
    Initialise Streamlit session state keys on first load.

    Session state persists across reruns within the same browser session,
    which lets us keep conversation history and avoid re-initialising heavy
    objects (like ChromaDB) on every Streamlit rerun.
    """
    if "messages" not in st.session_state:
        # Each message: {"role": "user"|"assistant", "content": str,
        #                 "under_the_hood": dict | None}
        st.session_state.messages = []

    if "store_ready" not in st.session_state:
        st.session_state.store_ready = False

    if "last_config_hash" not in st.session_state:
        st.session_state.last_config_hash = None

    # Track the active persona so we can detect switches
    if "active_persona" not in st.session_state:
        st.session_state.active_persona = DEFAULT_PERSONA


def _config_changed(cfg: AppConfig) -> bool:
    """Return True if the configuration has changed since the last ingest."""
    current_hash = hash(
        (cfg.chunk_size, cfg.chunk_overlap, cfg.collection_name)
    )
    return current_hash != st.session_state.last_config_hash


def _mark_config_stable(cfg: AppConfig) -> None:
    """Record the current configuration hash so we can detect future changes."""
    st.session_state.last_config_hash = hash(
        (cfg.chunk_size, cfg.chunk_overlap, cfg.collection_name)
    )


# =============================================================================
# Sidebar — Configuration
# =============================================================================

def render_sidebar() -> AppConfig:
    """
    Render the sidebar configuration panel and return an :class:`AppConfig`.

    The sidebar is the "control panel" of the RAG playground.  Every
    parameter that affects the pipeline is exposed here as an interactive
    widget, making it easy to experiment with different settings and
    immediately see their effect on retrieval quality.

    Returns
    -------
    AppConfig
        Configuration object built from the current sidebar widget values.
    """
    with st.sidebar:
        st.title("🍁 RAG Playground")

        # --- Persona Selector (at the very top of the sidebar) ---
        st.subheader("🎭 Persona")
        persona_names = list(PERSONAS.keys())
        selected_persona = st.selectbox(
            "Select Persona",
            options=persona_names,
            index=persona_names.index(st.session_state.active_persona),
            help="Switch between domain-specific assistants. Each persona "
                 "uses its own document collection and system prompt.",
        )

        # Detect persona change — reset chat history and store readiness
        if selected_persona != st.session_state.active_persona:
            st.session_state.active_persona = selected_persona
            st.session_state.messages = []
            st.session_state.store_ready = False
            st.session_state.last_config_hash = None

        persona_cfg = PERSONAS[st.session_state.active_persona]
        st.caption(persona_cfg["ui_title"])

        st.divider()

        # --- API Key ---
        st.subheader("🔑 OpenAI API Key")
        api_key_from_env = os.getenv("OPENAI_API_KEY", "")
        if api_key_from_env:
            st.success("API key loaded from environment.", icon="✅")
            api_key = api_key_from_env
        else:
            api_key = st.text_input(
                "Enter your OpenAI API key",
                type="password",
                placeholder="sk-...",
                help="Your key is never stored or logged.",
            )
            if api_key:
                st.success("API key set.", icon="✅")
            else:
                st.warning("API key required to use the assistant.", icon="⚠️")

        st.divider()

        # --- Chunking Strategy ---
        st.subheader("✂️ Chunking Strategy")
        chunking_strategy = st.radio(
            "Strategy",
            options=["Recursive Character (Recommended)", "Fixed-Size Overlap"],
            key="chunking_strategy_radio",
            help=(
                "**Recursive Character**: Prefers natural language boundaries "
                "(paragraphs → sentences → words). Best for legal/tax documents.\n\n"
                "**Fixed-Size Overlap**: Simple sliding window. Fast but may "
                "cut sentences in half."
            ),
        )

        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=3000,
            value=1000,
            step=100,
            help=(
                "Target character count per chunk. "
                "Larger chunks → more context per retrieval but coarser matching. "
                "Smaller chunks → more precise matching but less context."
            ),
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=min(500, chunk_size - 50),
            value=min(150, chunk_size // 6),
            step=25,
            help=(
                "Characters shared between consecutive chunks. "
                "Prevents important sentences from being split across chunk boundaries."
            ),
        )

        st.divider()

        # --- Retrieval ---
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=15,
            value=5,
            help=(
                "Number of document chunks to retrieve per query. "
                "More chunks → more context for the LLM but higher token cost."
            ),
        )

        st.divider()

        # --- Danger zone ---
        st.subheader("⚠️ Database")
        if st.button("🗑️ Clear Vector Store", type="secondary", use_container_width=True):
            st.session_state.store_ready = False
            st.session_state.last_config_hash = None
            st.session_state.messages = []
            st.warning("Vector store will be cleared on next ingest.")

        st.divider()
        st.caption(
            "**Universal RAG Playground** · "
            "[GitHub](https://github.com/gmjadeja/rag_demo)"
        )

    return AppConfig(
        openai_api_key=api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        collection_name=persona_cfg["collection_name"],
    )


# =============================================================================
# Document Ingestion Panel
# =============================================================================

def render_ingestion_panel(cfg: AppConfig) -> None:
    """
    Render the document upload and ingestion section.

    Users can either upload their own CRA PDF/text files or use the bundled
    sample Canadian tax document.  On submission, the processor chunks the
    document and the vector store embeds and persists the chunks.

    Parameters
    ----------
    cfg:
        Current application configuration (chunk size, overlap, etc.).
    """
    st.header("📄 Document Ingestion")

    # Resolve persona-specific labels for the UI
    persona_cfg = PERSONAS[st.session_state.active_persona]

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_files = st.file_uploader(
            f"Upload documents for {persona_cfg['ui_title']} (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help=(
                "Upload documents relevant to the selected persona. "
                "PDF and plain text are supported."
            ),
        )

    with col2:
        use_sample = st.checkbox(
            "Use bundled sample document",
            value=not bool(uploaded_files),
            help="Load the included CRA T4 Guide excerpt for a quick demo.",
        )
        sample_meta = {
            "doc_type": "T4 Guide",
            "tax_year": "2024",
            "source_url": "https://www.canada.ca/en/revenue-agency.html",
        }

    if st.button("🚀 Ingest Documents", type="primary", use_container_width=False):
        if not cfg.is_configured():
            st.error("Please enter your OpenAI API key in the sidebar first.")
            return

        files_to_process: list[tuple[str | Path, dict]] = []

        # Bundled sample document
        if use_sample:
            sample_path = Path(__file__).parent / "data" / "cra_sample.txt"
            if sample_path.exists():
                files_to_process.append((sample_path, sample_meta))
            else:
                st.warning("Sample document not found at data/cra_sample.txt")

        # User-uploaded files
        if uploaded_files:
            for uf in uploaded_files:
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uf.read())
                    tmp_path = Path(tmp.name)
                files_to_process.append(
                    (tmp_path, {"source": uf.name, "doc_type": "Uploaded"})
                )

        if not files_to_process:
            st.warning("No documents selected. Enable the sample or upload a file.")
            return

        _run_ingestion(cfg, files_to_process)


def _run_ingestion(
    cfg: AppConfig,
    files: list[tuple[str | Path, dict]],
) -> None:
    """
    Execute the ingestion pipeline and update session state.

    Parameters
    ----------
    cfg:
        Application configuration.
    files:
        List of ``(path, metadata)`` tuples to ingest.
    """
    # Build chunker based on sidebar selection (stored via session_state trick)
    # We re-read the radio value from session state
    strategy = st.session_state.get("chunking_strategy_radio", "Recursive Character (Recommended)")

    if "Fixed-Size" in strategy:
        chunker = FixedSizeChunker(cfg.chunk_size, cfg.chunk_overlap)
    else:
        chunker = RecursiveCharacterChunker(cfg.chunk_size, cfg.chunk_overlap)

    processor = DocumentProcessor(chunker)

    with st.spinner("Ingesting documents — embedding chunks with OpenAI..."):
        try:
            store = ChromaVectorStore(cfg)

            # Clear existing data if config changed to avoid stale index
            if _config_changed(cfg) and store.count() > 0:
                store.delete_collection()

            all_chunks = []
            for file_path, meta in files:
                chunks = processor.process_file(file_path, meta)
                all_chunks.extend(chunks)

            if not all_chunks:
                st.error("No text could be extracted from the provided files.")
                return

            store.add_chunks(all_chunks)
            _mark_config_stable(cfg)
            st.session_state.store_ready = True

            st.success(
                f"✅ Ingested **{len(all_chunks)} chunks** from "
                f"**{len(files)} document(s)**. Ready to answer questions!"
            )
            st.info(
                f"📊 Chunk size: **{cfg.chunk_size}** chars | "
                f"Overlap: **{cfg.chunk_overlap}** chars | "
                f"Chunker: **{type(chunker).__name__}**"
            )
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
            logger.exception("Ingestion error")


# =============================================================================
# Chat Interface
# =============================================================================

def render_chat(cfg: AppConfig) -> None:
    """
    Render the main chat interface with "Under the Hood" expanders.

    For each assistant response, an expander reveals:

    1.  The exact document chunks retrieved (with similarity scores and metadata).
    2.  The complete prompt sent to the LLM.

    This transparency is the educational core of the playground — users can
    see *why* the model gave a particular answer and identify retrieval or
    prompt quality issues.

    Parameters
    ----------
    cfg:
        Application configuration used to build the pipeline components.
    """
    persona_cfg = PERSONAS[st.session_state.active_persona]
    st.header(f"💬 Chat with the {persona_cfg['ui_title']}")

    if not st.session_state.store_ready:
        st.info(
            "👆 Please ingest at least one document above before asking questions.",
            icon="ℹ️",
        )
        return

    if not cfg.is_configured():
        st.error("OpenAI API key is required. Please enter it in the sidebar.")
        return

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show "Under the Hood" for assistant messages that have debug data
            if msg["role"] == "assistant" and msg.get("under_the_hood"):
                _render_under_the_hood(msg["under_the_hood"])

    # Chat input
    if user_input := st.chat_input("Ask a question…"):
        # Add user message to history
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "under_the_hood": None}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                response_data = _run_rag_pipeline(user_input, cfg)

            if response_data is None:
                return

            answer = response_data["answer"]
            st.markdown(answer)
            _render_under_the_hood(response_data)

        # Persist to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "under_the_hood": response_data,
            }
        )


def _run_rag_pipeline(
    query: str,
    cfg: AppConfig,
) -> dict | None:
    """
    Execute the full RAG pipeline for a single query and return debug data.

    Pipeline steps
    --------------
    1.  Retrieve relevant chunks from ChromaDB.
    2.  Format the chunks into a context string.
    3.  Generate an answer using the OpenAI Chat API.
    4.  Return the answer + all intermediate data for the UI.

    Parameters
    ----------
    query:
        User's question.
    cfg:
        Application configuration.

    Returns
    -------
    dict | None
        Dictionary with keys: ``answer``, ``retrieved_chunks``,
        ``context_string``, ``full_prompt``, ``model``, ``usage``.
        Returns ``None`` if the pipeline raises an exception.
    """
    try:
        store = ChromaVectorStore(cfg)
        retriever = Retriever(store, cfg)

        # Resolve the active persona's system-prompt template
        persona_cfg = PERSONAS[st.session_state.active_persona]
        generator = RAGGenerator(
            cfg,
            system_prompt_template=persona_cfg["system_prompt"],
        )

        # --- Step 1: Retrieve ---
        results = retriever.retrieve(query, top_k=cfg.top_k)
        if not results:
            no_context = "No relevant context found in the document collection."
            return {
                "answer": (
                    "I cannot find information about that in the provided "
                    "documents. Please ingest relevant documents for the "
                    "selected persona and try again."
                ),
                "retrieved_chunks": [],
                "context_string": no_context,
                "full_prompt": f"[No context retrieved]\nUser: {query}",
                "model": cfg.chat_model,
                "usage": {},
            }

        # --- Step 2: Format context ---
        context = retriever.format_context(results)

        # --- Step 3: Generate ---
        gen_response = generator.generate(query=query, context=context)

        return {
            "answer": gen_response.answer,
            "retrieved_chunks": results,
            "context_string": context,
            "full_prompt": gen_response.full_prompt,
            "model": gen_response.model,
            "usage": gen_response.usage,
        }

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        logger.exception("RAG pipeline error")
        return None


def _render_under_the_hood(data: dict) -> None:
    """
    Render the "Under the Hood" expander for a single assistant response.

    This is the signature educational feature of the playground.  It demystifies
    the RAG process by showing developers exactly what happened behind the scenes
    to produce the visible answer.

    Parameters
    ----------
    data:
        Dictionary from :func:`_run_rag_pipeline` containing retrieved chunks,
        the context string, and the full prompt.
    """
    with st.expander("🔍 Under the Hood — See how this answer was generated"):

        # --- Tab layout for clean organisation ---
        tab_chunks, tab_context, tab_prompt, tab_usage = st.tabs(
            ["📚 Retrieved Chunks", "🧩 Formatted Context", "📝 Full Prompt", "📊 Usage"]
        )

        # ---- Tab 1: Retrieved Chunks ----------------------------------------
        with tab_chunks:
            st.markdown(
                "These are the exact CRA document chunks that were retrieved "
                "from the vector store based on **semantic similarity** to your query."
            )
            chunks = data.get("retrieved_chunks", [])
            if not chunks:
                st.info("No chunks were retrieved for this query.")
            else:
                for rank, chunk in enumerate(chunks, start=1):
                    score = chunk.score
                    source = chunk.metadata.get("source", "Unknown")
                    tax_year = chunk.metadata.get("tax_year", "N/A")
                    doc_type = chunk.metadata.get("doc_type", "N/A")

                    # Colour-code by score
                    if score >= 0.85:
                        badge = "🟢 High relevance"
                    elif score >= 0.70:
                        badge = "🟡 Medium relevance"
                    else:
                        badge = "🔴 Low relevance"

                    with st.container(border=True):
                        col_rank, col_score, col_badge = st.columns([1, 2, 3])
                        col_rank.metric("Rank", f"#{rank}")
                        col_score.metric("Similarity Score", f"{score:.4f}")
                        col_badge.markdown(f"**{badge}**")

                        st.markdown(
                            f"**Source:** `{source}` · "
                            f"**Tax Year:** `{tax_year}` · "
                            f"**Type:** `{doc_type}`"
                        )
                        st.text_area(
                            "Chunk Text",
                            value=chunk.text,
                            height=120,
                            disabled=True,
                            key=f"chunk_{rank}_{id(data)}",
                        )

        # ---- Tab 2: Formatted Context ----------------------------------------
        with tab_context:
            st.markdown(
                "This is the formatted context string that was injected into "
                "the system prompt.  It concatenates all retrieved chunks with "
                "their metadata headers."
            )
            st.code(
                data.get("context_string", ""),
                language="text",
            )

        # ---- Tab 3: Full Prompt ---------------------------------------------
        with tab_prompt:
            st.markdown(
                "This is the **complete prompt** sent to the LLM — both the "
                "system message (with context and constraints) and your user message. "
                "The system prompt explicitly forbids hallucinating US tax laws."
            )
            st.code(
                data.get("full_prompt", ""),
                language="text",
            )

        # ---- Tab 4: Usage ---------------------------------------------------
        with tab_usage:
            usage = data.get("usage", {})
            model = data.get("model", "N/A")
            st.markdown(f"**Model:** `{model}`")
            if usage:
                col_p, col_c, col_t = st.columns(3)
                col_p.metric("Prompt Tokens", usage.get("prompt_tokens", 0))
                col_c.metric("Completion Tokens", usage.get("completion_tokens", 0))
                col_t.metric("Total Tokens", usage.get("total_tokens", 0))
            else:
                st.info("Token usage data not available.")


# =============================================================================
# Main entrypoint
# =============================================================================

def main() -> None:
    """
    Main application entrypoint.

    Streamlit reruns this entire function on every user interaction.
    Session state (``st.session_state``) is used to persist data across
    reruns without re-initialising expensive objects.
    """
    _init_session_state()

    # Build configuration from sidebar
    cfg = render_sidebar()

    # Hero section — dynamically reflects the active persona
    persona_cfg = PERSONAS[st.session_state.active_persona]
    st.title(f"🍁 Universal RAG Playground — {persona_cfg['ui_title']}")
    st.markdown(
        """
        **A production-ready, educational Retrieval-Augmented Generation (RAG) system.**

        This playground demonstrates every step of the RAG pipeline.
        Select a **Persona** in the sidebar to switch between domain-specific
        assistants (e.g. Canadian Tax or Immigration). Upload documents,
        configure the pipeline, and explore the "Under the Hood" expanders to
        see exactly what happens at each stage.

        ---
        """
    )

    # Architecture overview
    with st.expander("📐 RAG Architecture Overview", expanded=False):
        st.markdown(
            """
            The Universal RAG Playground implements a clean, production-ready
            RAG architecture with full separation of concerns:

            ```
            User Question
                ↓
            [Retriever] → Embed query → ChromaDB ANN search
                ↓
            Retrieved Chunks (with similarity scores & metadata)
                ↓
            [Generator] → Inject chunks into system prompt
                ↓
            OpenAI Chat API
                ↓
            Grounded Answer
            ```

            **Modules:**
            - `processor.py` — Loads PDF/text files and chunks them
            - `vector_store.py` — Embeds chunks and stores/queries ChromaDB
            - `retriever.py` — Retrieves and ranks relevant chunks
            - `generator.py` — Constructs grounded prompts and calls the LLM
            - `app.py` — This Streamlit UI
            """
        )

    # Ingestion panel
    render_ingestion_panel(cfg)

    st.divider()

    # Chat interface
    render_chat(cfg)


if __name__ == "__main__":
    main()
