# 🍁 Universal RAG Playground — Canadian Tax Assistant

A **production-ready, educational Retrieval-Augmented Generation (RAG)** system built with Python, Streamlit, ChromaDB, and the OpenAI API.

The default implementation is a **Canadian Tax Assistant** that ingests CRA documents (T4 guides, tax folios, Income Tax Act excerpts) and answers questions without hallucinating US tax law.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Educational UI** | Every response shows retrieved chunks, similarity scores, and the full LLM prompt |
| **Two Chunking Strategies** | Recursive Character (best for legal docs) + Fixed-Size Overlap |
| **Metadata Filtering** | Filter by tax year, document type, or any custom field |
| **Local Vector DB** | ChromaDB — no server, no Docker, persists to disk |
| **Clean Architecture** | Abstract Base Classes, Dependency Injection, Separation of Concerns |
| **Hallucination Guard** | Strict system prompt forbids US tax law references and unsupported answers |

---

## 🗂️ Project Structure

```
rag_demo/
├── app.py              # Streamlit UI (main entry point)
├── processor.py        # Document loading & chunking
├── vector_store.py     # ChromaDB embedding & storage
├── retriever.py        # Semantic search & result ranking
├── generator.py        # Prompt construction & LLM call
├── config.py           # Shared configuration & constants
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── data/
    └── cra_sample.txt  # Bundled Canadian tax sample document
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/gmjadeja/rag_demo.git
cd rag_demo
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 RAG Pipeline Architecture

```
User Question
    ↓
[Retriever]  →  Embed query  →  ChromaDB ANN search
    ↓
Retrieved Chunks  (with similarity scores & metadata)
    ↓
[Generator]  →  Inject chunks into system prompt  →  OpenAI Chat API
    ↓
Grounded Answer  (with "Under the Hood" transparency panel)
```

### Module Breakdown

| Module | Responsibility |
|---|---|
| `processor.py` | Loads `.txt`/`.pdf` files, splits into chunks using pluggable strategies |
| `vector_store.py` | Embeds chunks via OpenAI, stores/queries ChromaDB, converts L2 → cosine similarity |
| `retriever.py` | Orchestrates retrieval, applies score filtering, formats context for the generator |
| `generator.py` | Injects context into strict system prompt, calls GPT-4o-mini, returns full prompt for UI |
| `config.py` | Single source of truth for all tuneable parameters |
| `app.py` | Streamlit UI: sidebar config, file upload, chat interface, "Under the Hood" expanders |

---

## 🔍 "Under the Hood" Expander

For every assistant response, click **🔍 Under the Hood** to see:

1. **Retrieved Chunks** — The exact CRA document passages retrieved, with similarity scores colour-coded (🟢 ≥0.85, 🟡 0.70–0.85, 🔴 <0.70) and provenance metadata.
2. **Formatted Context** — The context string injected into the system prompt.
3. **Full Prompt** — The complete system + user messages sent to the LLM.
4. **Usage** — Token counts (prompt, completion, total).

---

## ⚙️ Configuration

All parameters are adjustable in the Streamlit sidebar:

| Parameter | Default | Description |
|---|---|---|
| Chunking Strategy | Recursive Character | Strategy used to split documents |
| Chunk Size | 1 000 chars | Target characters per chunk |
| Chunk Overlap | 150 chars | Overlap between consecutive chunks |
| Top-K Chunks | 5 | Number of chunks retrieved per query |

---

## 🔒 Hallucination Prevention

The system prompt explicitly:
- Restricts answers to the provided CRA document context only
- Forbids references to US tax law (IRS, 401(k), Roth IRA, W-2, etc.)
- Instructs the model to respond with a specific refusal message when the answer is not in the context
- Uses `temperature=0.0` for maximum factual precision

---

## 🧪 Tech Stack

- **Python 3.11+**
- **Streamlit** — Interactive UI
- **ChromaDB** — Local vector database (HNSW index, cosine distance)
- **OpenAI API** — `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation
- **pypdf** — PDF text extraction

---

## 📄 License

MIT
