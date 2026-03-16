"""
generator.py — Augmented Generation
=====================================

Purpose
-------
Construct the final prompt by combining the user's question with the
retrieved CRA document chunks, then call the OpenAI Chat Completions API to
produce a grounded, citation-aware answer.

The Augmented Generation step is the *AG* in RAG.  Its two critical
responsibilities are:

1.  **Prompt engineering** — wrap the retrieved context and user question in
    a carefully designed system prompt that constrains the model's behaviour.
2.  **Hallucination prevention** — the system prompt explicitly forbids the
    model from drawing on any knowledge outside the provided context,
    particularly US tax law, which is the most common source of confusion
    when asking about Canadian taxes.

Why a dedicated generator module?
-----------------------------------
The generator encapsulates all LLM-facing logic.  Separating it from the
retriever and vector store means:

*   Prompt templates can be updated without touching retrieval code.
*   The ``GeneratorResponse`` dataclass carries both the final answer *and*
    the full prompt sent to the LLM, which the Streamlit UI displays in the
    "Under the Hood" expander.
*   Swapping GPT-4o for Claude or a local LLaMA model requires only a new
    ``BaseGenerator`` subclass.

System prompt design rationale
-------------------------------
The system prompt follows a *constrained grounding* pattern:

1.  **Role** — establishes the model as a specialist in *Canadian* tax law.
2.  **Ground truth** — injects the retrieved CRA chunks as the sole source
    of facts.
3.  **Strict constraints** — explicitly forbids hallucination, US tax law
    references, and off-topic responses.
4.  **Fallback instruction** — tells the model exactly what to say when the
    context does not contain a sufficient answer, preventing fabricated
    responses.

Usage example
-------------
::

    from generator import RAGGenerator
    from retriever import Retriever
    from vector_store import ChromaVectorStore
    from config import AppConfig

    cfg = AppConfig()
    store = ChromaVectorStore(cfg)
    retriever = Retriever(store, cfg)
    generator = RAGGenerator(cfg)

    results = retriever.retrieve("What is the RRSP deduction limit for 2024?")
    context = retriever.format_context(results)
    response = generator.generate(
        query="What is the RRSP deduction limit for 2024?",
        context=context,
    )
    print(response.answer)
    print("\\n--- Prompt sent to LLM ---")
    print(response.full_prompt)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import AppConfig

logger = logging.getLogger(__name__)


# =============================================================================
# System prompt template
# =============================================================================

#: The system prompt sent to the LLM on every request.
#:
#: ``{context}`` is replaced with the formatted retrieval results.
#: The strict constraints prevent the model from drawing on its pre-training
#: knowledge and fabricating plausible-sounding but incorrect tax rules.
SYSTEM_PROMPT_TEMPLATE = """\
You are a highly specialised Canadian Tax Assistant with deep expertise in the \
Canada Revenue Agency (CRA) tax regulations, the Income Tax Act (Canada), and \
related CRA publications.

YOUR SOLE SOURCE OF TRUTH
==========================
You must answer ONLY using the context excerpts provided below. These excerpts \
have been retrieved from official CRA documents and are the only information \
you are permitted to use when formulating your answer.

RETRIEVED CRA DOCUMENT CONTEXT
================================
{context}

STRICT RULES — READ CAREFULLY
==============================
1. ONLY answer based on the context above. Do NOT use any outside knowledge.
2. NEVER reference, mention, or confuse Canadian tax rules with US tax rules \
   (IRS, 401(k), Roth IRA, W-2, etc.). These are completely different systems.
3. If the answer to the question is NOT present in the provided context, you \
   MUST respond with exactly: "I cannot find information about that in the \
   provided CRA documents. Please consult a qualified Canadian tax professional \
   or visit the CRA website directly."
4. When you cite a fact, always mention its source document if it appears in \
   the context metadata.
5. Do NOT speculate, extrapolate, or fill in gaps with assumptions.
6. Be concise and precise. Tax law is nuanced — do not over-simplify.
"""


# =============================================================================
# Public data contract
# =============================================================================


@dataclass
class GeneratorResponse:
    """
    The complete output of a single RAG generation cycle.

    Attributes
    ----------
    answer:
        The final natural-language answer produced by the LLM.
    full_prompt:
        The complete prompt (system + user messages) sent to the LLM.
        Exposed in the Streamlit "Under the Hood" expander so users can
        understand exactly what the model received.
    model:
        The OpenAI model that produced the answer.
    usage:
        Token usage statistics from the OpenAI API response
        (``{"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}``).
    """

    answer: str
    full_prompt: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Abstract base class
# =============================================================================


class BaseGenerator:
    """
    Abstract base class for answer generators.

    Subclasses must implement :meth:`generate`.
    """

    def generate(self, query: str, context: str) -> GeneratorResponse:
        """
        Generate an answer for *query* grounded in *context*.

        Parameters
        ----------
        query:
            The user's natural-language question.
        context:
            Pre-formatted string of retrieved document chunks (as produced by
            :meth:`~retriever.Retriever.format_context`).

        Returns
        -------
        GeneratorResponse
            The answer, full prompt, model name, and token usage.
        """
        raise NotImplementedError


# =============================================================================
# OpenAI-backed generator
# =============================================================================


class RAGGenerator(BaseGenerator):
    """
    Answer generator backed by the OpenAI Chat Completions API.

    Prompt construction
    -------------------
    The prompt uses the standard two-message structure:

    1.  **System message** — Establishes the assistant's persona, injects the
        retrieved context, and lays out the strict behavioural constraints.
        This is the most important part of hallucination prevention: the model
        is told *what it knows* (the context) and *what it must not do*
        (reference US tax law or answer without evidence).

    2.  **User message** — Contains only the raw user question.  Keeping the
        question separate from the system message makes it easier to swap
        prompt templates independently.

    Why we don't use a single "human turn" prompt
    -----------------------------------------------
    Many RAG tutorials concatenate everything into one big prompt.  The
    system/user split is better because:

    *   OpenAI fine-tunes its models to pay extra attention to the system
        message for behavioural constraints.
    *   It mirrors production best practices and makes prompt injection harder.

    Parameters
    ----------
    config:
        Application configuration (API key, chat model name).
    """

    def __init__(self, config: AppConfig) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.chat_model

    def generate(self, query: str, context: str) -> GeneratorResponse:
        """
        Build a grounded prompt and call the OpenAI API.

        Step-by-step walkthrough
        ------------------------
        1.  **Context injection** — Insert the retrieved CRA chunks into
            ``SYSTEM_PROMPT_TEMPLATE``.
        2.  **Message list** — Construct the ``[system, user]`` message list
            that the Chat Completions API expects.
        3.  **API call** — Send the messages and capture the response.
        4.  **Result packaging** — Wrap the answer, full prompt representation,
            model name, and token usage into a :class:`GeneratorResponse`.

        Parameters
        ----------
        query:
            The user's question.
        context:
            Formatted string from :meth:`~retriever.Retriever.format_context`.

        Returns
        -------
        GeneratorResponse
            Complete generation result including the full prompt for
            the educational UI.
        """
        system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        # Represent the full prompt as a human-readable string for the UI
        full_prompt = (
            "=== SYSTEM MESSAGE ===\n"
            + system_content
            + "\n\n=== USER MESSAGE ===\n"
            + query
        )

        logger.info(
            "Calling OpenAI Chat API (model=%s, ~%d chars in system prompt)",
            self._model,
            len(system_content),
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.0,    # Zero temperature for maximum factual precision
            max_tokens=1024,
        )

        answer = response.choices[0].message.content or ""

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info(
            "Generation complete. Tokens used: %s", usage
        )

        return GeneratorResponse(
            answer=answer,
            full_prompt=full_prompt,
            model=self._model,
            usage=usage,
        )
