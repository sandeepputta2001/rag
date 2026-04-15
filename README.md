# TechCorp AI Assistant — RAG-Powered Chat Application

An interactive question-answering system built on **Retrieval-Augmented Generation (RAG)**. It lets employees ask natural language questions about company policies, products, and engineering practices, and get accurate answers grounded in TechCorp's internal documents.

---

## Table of Contents

1. [How to Run](#how-to-run)
2. [How to Use](#how-to-use)
3. [High-Level Design (HLD)](#high-level-design-hld)
4. [Low-Level Design (LLD)](#low-level-design-lld)
5. [Project Structure](#project-structure)
6. [API Reference](#api-reference)

---

## How to Run

### Prerequisites

- Python 3.10+
- pip (system-level)

### 1. Install Dependencies

```bash
pip3 install flask chromadb sentence-transformers anthropic --break-system-packages
```

> If you are using a virtual environment, activate it first and omit `--break-system-packages`.

### 2. (Optional) Enable Claude AI Responses

Without an API key the app uses a keyword-extraction fallback. For full LLM-quality answers, export your Anthropic key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Start the Application

```bash
/usr/bin/python3 app.py
```

On first run the app automatically ingests all TechCorp documents into ChromaDB. Subsequent restarts reuse the persisted database.

```
============================================================
Starting TechCorp AI Assistant
============================================================
[INIT] Loading RAG components...
[INIT] Vector engine ready
[INIT] Chat engine ready
[INIT] Document processor ready
 * Running on http://0.0.0.0:5252
```

### 4. Open the Chat UI

Navigate to `http://localhost:5252` in your browser.

### Running the Standalone Test Scripts

| Script | Purpose |
|---|---|
| `python3 test_chunking.py` | Verify the text-chunking logic |
| `python3 test_embeddings.py` | Verify sentence-transformer embeddings |
| `python3 init_vectordb.py` | Create / reset the ChromaDB collection |
| `python3 ingest_documents.py` | Re-ingest documents from `/root/techcorp-docs` |
| `python3 test_search.py` | Run semantic search queries against the DB |
| `python3 test_rag_pipeline.py` | End-to-end pipeline smoke test |

---

## How to Use

### Chat Interface

Open `http://localhost:5252` and type any question about TechCorp into the input box. Press **Enter** or click **Send**.

**Example questions:**

- _"What is the pet policy?"_
- _"How many days of remote work are allowed?"_
- _"What health benefits does TechCorp offer?"_
- _"Tell me about CloudSync Pro pricing."_
- _"What tech stack does engineering use?"_

Each response includes:
- **Answer** — grounded in retrieved document chunks
- **Sources** — category and filename of the documents used
- **Confidence** — cosine-similarity score from the vector search

### REST API

```bash
# Ask a question
curl -X POST http://localhost:5252/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the employee benefits?"}'

# System status
curl http://localhost:5252/api/status
```

---

## High-Level Design (HLD)

```
┌─────────────────────────────────────────────────────────────────┐
│                        User / Browser                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP (port 5252)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Web Server  (app.py)                   │
│                                                                 │
│   GET  /              → Chat UI (HTML)                          │
│   POST /api/chat      → RAG answer                              │
│   POST /api/chat/stream → Streaming RAG answer (SSE)            │
│   GET  /api/status    → System health                           │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐       ┌──────────────────────┐
│   ChatEngine     │       │   DocumentProcessor  │
│                  │       │                      │
│  Orchestrates    │       │  Loads & chunks      │
│  Retrieval  +    │       │  TechCorp docs on    │
│  Generation      │       │  first startup       │
└──────┬───────────┘       └──────────┬───────────┘
       │                              │
       │ search(query)                │ add_documents(chunks)
       │                              │
       └──────────────┬───────────────┘
                      ▼
         ┌────────────────────────┐
         │     VectorEngine       │
         │                        │
         │  Embeds text with      │
         │  all-MiniLM-L6-v2     │
         │  Stores/queries        │
         │  ChromaDB (cosine)     │
         └──────────┬─────────────┘
                    │
                    ▼
         ┌────────────────────────┐
         │   ChromaDB (on-disk)   │
         │   ./chroma_db/         │
         │   collection:          │
         │   techcorp_docs        │
         └────────────────────────┘
```

### Data Flow Summary

```
User Question
     │
     ▼
[1] Embed question → 384-dim vector  (all-MiniLM-L6-v2)
     │
     ▼
[2] Vector search → top-3 chunks     (ChromaDB cosine similarity)
     │
     ▼
[3] Build context string from chunks
     │
     ▼
[4a] ANTHROPIC_API_KEY set?
     ├── YES → Claude Haiku generates answer from context
     └── NO  → Keyword-extraction fallback from context
     │
     ▼
[5] Return answer + sources + confidence score → browser
```

### Knowledge Base Contents

| Category | Documents | Topics |
|---|---|---|
| `hr` | pet_policy.md, remote_work.md, benefits.md | Office rules, WFH policy, insurance, 401k, PTO |
| `products` | cloudsync_pro.md, techassist_ai.md | Pricing, features, SLAs |
| `engineering` | tech_stack.md | Languages, infra, CI/CD, code review standards |

---

## Low-Level Design (LLD)

### `VectorEngine` — `core/vector_engine.py`

Responsible for all interactions with the vector store.

```
VectorEngine
├── __init__(db_path, collection_name)
│     └── loads SentenceTransformer('all-MiniLM-L6-v2')
│         creates/opens ChromaDB PersistentClient
│         gets/creates collection with cosine distance
│
├── is_initialized() → bool
│     └── returns collection.count() > 0
│
├── add_documents(chunks: list[str], metadatas: list[dict])
│     ├── encode all chunks → float32 embeddings (batch)
│     ├── auto-generate sequential IDs
│     └── collection.add(ids, embeddings, documents, metadatas)
│
├── search(query: str, n_results=3) → dict
│     ├── encode query → embedding
│     └── collection.query() → {documents, metadatas, distances}
│
└── get_stats() → dict
      └── {total_documents, total_chunks, last_updated}
```

**Embedding model:** `all-MiniLM-L6-v2`
- Output dimension: 384
- Max input tokens: 256
- Optimised for semantic similarity tasks

**ChromaDB collection settings:**
- Distance metric: cosine
- Storage: persistent on disk at `./chroma_db/`
- Telemetry: disabled

---

### `ChatEngine` — `core/chat_engine.py`

Orchestrates the Retrieve → Augment → Generate pipeline.

```
ChatEngine
├── __init__(vector_engine)
│     └── _setup_llm()
│           checks ANTHROPIC_API_KEY env var
│           if set → initialises anthropic.Anthropic client
│
├── get_response(question: str) → dict
│     ├── vector_engine.search(question, n_results=3)
│     ├── build context = "\n\n".join(top docs)
│     ├── compute per-source confidence: (1 - cosine_distance) × 100
│     ├── _generate_with_claude(question, context)   [if API key]
│     │     └── claude-haiku-4-5-20251001, max_tokens=512
│     │         system: "Answer only from provided context"
│     └── _generate_from_context(question, context)  [fallback]
│           keyword pattern-match → extract relevant lines
│           returns "Based on TechCorp documents: ..."
│
└── returns {answer, sources: [{category, file, relevance}], confidence}
```

**Generation modes:**

| Mode | Trigger | Model |
|---|---|---|
| Claude (LLM) | `ANTHROPIC_API_KEY` is set | `claude-haiku-4-5-20251001` |
| Fallback | No API key | Keyword-extraction from retrieved chunks |

---

### `DocumentProcessor` — `core/document_processor.py`

Handles document ingestion into the vector store.

```
DocumentProcessor
├── __init__(vector_engine)
│
└── process_all_documents() → {documents: int, chunks: int}
      for each (category, filename, content) in TECHCORP_DOCS:
          chunks = chunk_text(content, size=500, overlap=100)
          metadatas = [{file, category}] × len(chunks)
          vector_engine.add_documents(chunks, metadatas)
```

**Chunking strategy:**

```
chunk_text(text, size=500, overlap=100)

text:  [----chunk_1(500)----]
                  [----chunk_2(500)----]
       |<-- 400 -->|<-- 100 overlap -->|
```

- Chunk size: 500 characters
- Overlap: 100 characters (preserves sentence context across chunk boundaries)
- Guarantees no chunk exceeds the model's 256-token limit

---

### `app.py` — Flask Application

```
Routes
├── GET  /                  → render templates/chat.html
├── POST /api/chat          → chat_engine.get_response(message)
│                              returns {response, sources, confidence, timestamp}
├── POST /api/chat/stream   → Server-Sent Events (SSE)
│                              streams response word-by-word (50ms delay)
│                              final event carries sources + confidence
└── GET  /api/status        → vector_engine.get_stats()
                               returns {status, documents, chunks, last_updated}

Startup sequence
1. Instantiate VectorEngine  (loads model + opens DB)
2. Instantiate ChatEngine    (checks for API key)
3. Instantiate DocumentProcessor
4. if not vector_engine.is_initialized():
       doc_processor.process_all_documents()
5. app.run(host='0.0.0.0', port=5252, debug=True)
```

---

### `templates/chat.html` — Frontend

- Pure HTML/CSS/JS (no framework dependency)
- Communicates with `/api/chat` via `fetch` (JSON)
- Streaming endpoint `/api/chat/stream` consumes SSE events: `start → token → sources → done`
- Suggestion chips for quick queries
- Auto-resize textarea, Enter-to-send

---

## Project Structure

```
rag_project/
├── app.py                      # Flask app, routes, startup
├── core/
│   ├── __init__.py
│   ├── vector_engine.py        # ChromaDB + sentence-transformers
│   ├── chat_engine.py          # RAG pipeline (retrieve → generate)
│   └── document_processor.py  # Document chunking & ingestion
├── templates/
│   └── chat.html               # Chat UI
├── chroma_db/                  # Persisted vector store (auto-created)
├── test_chunking.py            # Unit test: chunking logic
├── test_embeddings.py          # Unit test: embedding similarity
├── test_search.py              # Integration test: semantic search
├── test_rag_pipeline.py        # Integration test: full RAG pipeline
├── init_vectordb.py            # Helper: create/reset ChromaDB
├── ingest_documents.py         # Helper: ingest from /root/techcorp-docs
└── venv/                       # Python virtual environment
```

---

## API Reference

### `POST /api/chat`

**Request**
```json
{ "message": "What is the pet policy?" }
```

**Response**
```json
{
  "response": "Based on TechCorp documents: Employees may bring pets to the office on Fridays...",
  "sources": [
    { "category": "hr", "file": "pet_policy.md", "relevance": 74.8 }
  ],
  "confidence": 74.8,
  "timestamp": "2026-04-14T14:51:23.768091"
}
```

### `GET /api/status`

**Response**
```json
{
  "status": "operational",
  "documents": 1,
  "chunks": 9,
  "last_updated": "N/A"
}
```

### `POST /api/chat/stream`

Returns a `text/event-stream` of SSE events:

```
data: {"event": "start"}
data: {"event": "token", "content": "Based "}
data: {"event": "token", "content": "on "}
...
data: {"event": "sources", "sources": [...], "confidence": 74.8}
data: {"event": "done"}
```
