# TechCorp AI Assistant

A Retrieval-Augmented Generation (RAG) chat application that lets employees ask natural language questions about TechCorp's internal policies, products, and engineering practices. The app finds the most relevant documents from a local knowledge base and generates grounded answers from them.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Knowledge Base](#knowledge-base)
- [How to Run](#how-to-run)
- [How to Use](#how-to-use)
- [API Reference](#api-reference)
- [Telemetry](#telemetry)
- [High-Level Design](#high-level-design)
- [Low-Level Design](#low-level-design)

---

## Project Structure

```
rag_project/
│
├── app.py                          # Flask server — routes and startup
│
├── core/
│   ├── telemetry.py                # OpenTelemetry setup (console or OTLP)
│   ├── vector_engine.py            # ChromaDB + sentence-transformers
│   ├── chat_engine.py              # RAG pipeline — retrieve → generate
│   └── document_processor.py      # Reads and chunks techcorp-docs/
│
├── templates/
│   └── chat.html                   # Browser chat UI (vanilla JS, no framework)
│
├── techcorp-docs/                  # Source documents (Markdown)
│   ├── engineering/
│   │   ├── on_call_runbook.md
│   │   └── tech_stack.md
│   ├── finance/
│   │   └── expense_policy.md
│   ├── hr/
│   │   ├── benefits.md
│   │   ├── code_of_conduct.md
│   │   ├── pet_policy.md
│   │   └── remote_work.md
│   ├── legal/
│   │   └── data_privacy_policy.md
│   └── products/
│       ├── cloudsync_pro.md
│       ├── databridge.md
│       └── techassist_ai.md
│
├── chroma_db/                      # Persisted vector store (auto-created on first run)
│
├── test_chunking.py                # Tests the text-chunking logic
├── test_embeddings.py              # Tests sentence-transformer embeddings
├── test_search.py                  # Runs semantic search queries against ChromaDB
├── test_rag_pipeline.py            # End-to-end RAG pipeline smoke test
├── init_vectordb.py                # Creates / resets the ChromaDB collection
└── ingest_documents.py             # Re-ingests documents into ChromaDB
```

---

## Knowledge Base

11 Markdown documents across 5 departments are pre-loaded into the vector store on first run.

| Category | File | Topics Covered |
|---|---|---|
| `hr` | `pet_policy.md` | Furry Fridays, eligible animals, designated areas, office mascot |
| `hr` | `remote_work.md` | Hybrid schedule, core hours, VPN, equipment stipends |
| `hr` | `benefits.md` | Medical/dental/vision, 401k, PTO, parental leave, learning budget |
| `hr` | `code_of_conduct.md` | Integrity, anti-harassment, conflict of interest, reporting |
| `engineering` | `tech_stack.md` | Languages, databases, Kubernetes, CI/CD, code review standards |
| `engineering` | `on_call_runbook.md` | Severity levels, escalation paths, incident playbooks, PIR process |
| `finance` | `expense_policy.md` | Travel limits, meal allowances, submission process, corporate cards |
| `legal` | `data_privacy_policy.md` | GDPR/CCPA, data retention periods, your rights, breach notification |
| `products` | `cloudsync_pro.md` | Features, pricing, SLA, system requirements |
| `products` | `techassist_ai.md` | AI support platform, integrations, pricing, onboarding |
| `products` | `databridge.md` | ETL connectors, pipeline modes, pricing |

---

## How to Run

### 1. Install dependencies

```bash
pip3 install flask chromadb sentence-transformers anthropic --break-system-packages
```

> If you use a virtual environment, activate it first and omit `--break-system-packages`.

### 2. (Optional) Enable Claude AI for better answers

Without an API key the app uses keyword extraction from retrieved chunks. With a key it calls Claude Haiku for fluent, context-aware answers.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Start the application

```bash
/usr/bin/python3 app.py
```

**What happens on first run:**

```
============================================================
Starting TechCorp AI Assistant
============================================================

[INIT] Loading RAG components...
[INIT] Telemetry ready (OpenTelemetry → stdout)
[INIT] Vector engine ready
[INIT] Chat engine ready
[INIT] Document processor ready
First run detected. Processing TechCorp documents...

  Processing category: engineering
    on_call_runbook.md: 11 chunks
    tech_stack.md: 13 chunks
  Processing category: finance
    expense_policy.md: 9 chunks
  Processing category: hr
    benefits.md: 12 chunks
    ...
  Total: 11 documents, 101 chunks ingested

 * Running on http://0.0.0.0:5252
```

Documents are embedded and saved to `chroma_db/`. Subsequent restarts skip ingestion and load instantly.

### 4. Open the chat UI

```
http://localhost:5252
```

---

### Running individual scripts

| Script | Command | Purpose |
|---|---|---|
| Chunking test | `python3 test_chunking.py` | Verify the 500-char / 100-overlap chunking logic |
| Embedding test | `python3 test_embeddings.py` | Check sentence-transformer similarity scores |
| Init vector DB | `python3 init_vectordb.py` | Create or reset the ChromaDB collection |
| Re-ingest docs | `python3 ingest_documents.py` | Load documents into ChromaDB from `techcorp-docs/` |
| Search test | `python3 test_search.py` | Run 3 semantic search queries and print results |
| Pipeline test | `python3 test_rag_pipeline.py` | End-to-end RAG smoke test with a sample question |

### Reset and re-ingest from scratch

```bash
rm -rf chroma_db/
/usr/bin/python3 app.py     # first-run detection triggers ingestion automatically
```

---

## How to Use

### Chat UI

Open `http://localhost:5252`. Type a question and press **Enter** or click **Send**.

Suggestion chips are shown on the welcome screen for quick starts:

- _What is the pet policy?_
- _How many remote work days are allowed?_
- _What benefits does TechCorp offer?_
- _Tell me about CloudSync Pro_
- _What is the tech stack used?_

Each response shows:

| Field | Meaning |
|---|---|
| **Answer** | Text grounded in retrieved document chunks |
| **Sources** | Category and filename of each chunk used |
| **Confidence** | Cosine similarity score (0–100) of the best-matching chunk |

### cURL examples

```bash
# Ask a question
curl -X POST http://localhost:5252/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the hotel expense limit when travelling?"}'

# Check system status
curl http://localhost:5252/api/status
```

---

## API Reference

### `POST /api/chat`

Ask a question. Returns a complete answer in one response.

#### Request

```json
{ "message": "What is the pet policy at TechCorp?" }
```

#### Response

```json
{
  "response": "Based on TechCorp documents: Employees may bring pets to the office on Fridays (Furry Fridays). Dogs must be well-behaved and up-to-date on vaccinations.",
  "sources": [
    { "category": "hr", "file": "pet_policy.md", "relevance": 74.8 }
  ],
  "confidence": 74.8,
  "timestamp": "2026-04-14T14:51:23.768091"
}
```

---

### `POST /api/chat/stream`

Same as `/api/chat` but streams the answer word-by-word using Server-Sent Events (SSE). Accepts the same JSON body: `{ "message": "..." }`.

#### Event stream

```
data: {"event": "start"}

data: {"event": "token", "content": "Based "}
data: {"event": "token", "content": "on "}
data: {"event": "token", "content": "TechCorp "}
...

data: {"event": "sources", "sources": [...], "confidence": 74.8}

data: {"event": "done"}
```

---

### `GET /api/status`

Returns the health and size of the vector store.

**Response**
```json
{
  "status": "operational",
  "documents": 1,
  "chunks": 101,
  "last_updated": "N/A"
}
```

---

## Telemetry

The app ships with two telemetry layers.

### 1. ChromaDB product telemetry (`anonymized_telemetry=True`)

Sends anonymous usage events (client version, Python version, OS, operation names — **never document content**) to ChromaDB's PostHog analytics. Helps the ChromaDB team prioritise development. Disable by setting `anonymized_telemetry=False` in `core/vector_engine.py`.

### 2. OpenTelemetry (OTEL) tracing

Every RAG operation emits a structured span. In development mode spans are printed to stdout. In production, point them at Jaeger, Grafana Tempo, or Honeycomb.

**Spans emitted:**

| Span name | Key attributes |
|---|---|
| `vector_engine.search` | `query`, `n_results_requested`, `n_results_returned`, `top_similarity_score`, `collection_size` |
| `vector_engine.encode` | `input_count`, `total_input_chars`, `embedding_dim`, `duration_ms` |
| `vector_engine.add_documents` | `chunk_count`, `category`, `embedding_dim`, `collection_size_after` |
| `vector_engine.get_stats` | `total_chunks` |

**Switch to a real tracing backend:**

```python
# app.py — change one argument
init_telemetry(service_name="techcorp-rag", console=False)
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # gRPC endpoint
```

---

## High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                       Browser                               │
│               http://localhost:5252                         │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                Flask Web Server  (app.py)                   │
│                                                             │
│  GET  /                  →  chat.html (UI)                  │
│  POST /api/chat          →  full JSON answer                │
│  POST /api/chat/stream   →  SSE word-by-word stream         │
│  GET  /api/status        →  system health                   │
└────────────┬──────────────────────────┬─────────────────────┘
             │                          │
             ▼                          ▼
   ┌──────────────────┐      ┌─────────────────────┐
   │   ChatEngine     │      │  DocumentProcessor  │
   │                  │      │                     │
   │  Orchestrates    │      │  Reads techcorp-docs │
   │  Retrieve +      │      │  Chunks + ingests   │
   │  Generate        │      │  on first startup   │
   └────────┬─────────┘      └──────────┬──────────┘
            │                           │
            │ search()                  │ add_documents()
            └──────────────┬────────────┘
                           ▼
              ┌────────────────────────┐
              │     VectorEngine       │
              │                        │
              │  Embeds with           │
              │  all-MiniLM-L6-v2     │
              │  (384-dim vectors)     │
              │                        │
              │  Stores/queries        │
              │  ChromaDB (cosine)     │
              └──────────┬─────────────┘
                         │
                         ▼
              ┌────────────────────────┐
              │   ChromaDB on disk     │
              │   ./chroma_db/         │
              │   101 chunks           │
              │   11 documents         │
              └────────────────────────┘
```

### RAG Data Flow

```
User types a question
         │
         ▼
[1] Embed question  ──►  384-dim float vector   (all-MiniLM-L6-v2, ~20 ms)
         │
         ▼
[2] Cosine search   ──►  Top-3 matching chunks  (ChromaDB HNSW index)
         │
         ▼
[3] Build context   ──►  Join chunks into a single string
         │
         ├─── ANTHROPIC_API_KEY set? ──YES──► Claude Haiku generates answer
         │
         └─────────────────────────── NO ───► Keyword extraction from context
         │
         ▼
[4] Return  { answer, sources, confidence }  ──►  Browser
```

---

## Low-Level Design

### `core/telemetry.py`

Initialises the global OpenTelemetry `TracerProvider` once before any component starts.

```
init_telemetry(service_name, console)
  │
  ├─ console=True  →  SimpleSpanProcessor + ConsoleSpanExporter
  │                   (each span printed immediately — good for local dev)
  │
  └─ console=False →  BatchSpanProcessor + OTLPSpanExporter
                      endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT (gRPC)
                      (non-blocking batch export — use in production)
```

---

### `core/vector_engine.py`

Wraps ChromaDB and the sentence-transformer model. Every public method is instrumented with an OTEL span.

```
VectorEngine.__init__
  └─ SentenceTransformer('all-MiniLM-L6-v2')   # 384-dim, 90M params
  └─ chromadb.PersistentClient(path='./chroma_db')
  └─ collection: techcorp_docs  (cosine distance, HNSW index)

add_documents(chunks, metadatas)
  └─ span: vector_engine.add_documents
       └─ _encode(chunks)          ← span: vector_engine.encode
       └─ collection.add(ids, embeddings, documents, metadatas)

search(query, n_results=3)
  └─ span: vector_engine.search
       └─ _encode([query])         ← span: vector_engine.encode
       └─ collection.query(query_embeddings, n_results)
       └─ returns {documents, metadatas, distances}

_encode(texts)                     # internal — called by search + add_documents
  └─ span: vector_engine.encode
       └─ model.encode(texts)      # returns float32 numpy arrays
       └─ .tolist()                # convert to Python lists for ChromaDB
```

**Embedding model specs:**

| Property | Value |
| --- | --- |
| Model | `all-MiniLM-L6-v2` |
| Output dimensions | 384 |
| Max input tokens | 256 |
| Approximate latency | 15–25 ms per query (CPU) |
| Distance metric | Cosine (stored in ChromaDB HNSW) |

---

### `core/chat_engine.py`

Implements the Retrieve → Augment → Generate pipeline.

```
ChatEngine.__init__(vector_engine)
  └─ checks ANTHROPIC_API_KEY env var
  └─ if set: initialises anthropic.Anthropic client

get_response(question) → {answer, sources, confidence}
  │
  ├─ vector_engine.search(question, n_results=3)
  │
  ├─ per-source confidence = (1 − cosine_distance) × 100
  │
  ├─ ANTHROPIC_API_KEY set?
  │      YES → _generate_with_claude(question, context)
  │               model: claude-haiku-4-5-20251001
  │               max_tokens: 512
  │               system prompt: "Answer only from provided context"
  │      NO  → _generate_from_context(question, context)
  │               keyword pattern-match → extract relevant lines
  │
  └─ overall_confidence = mean of per-source confidence scores
```

**Generation modes:**

| Mode | Trigger | Quality |
| --- | --- | --- |
| Claude Haiku | `ANTHROPIC_API_KEY` is set | Fluent, context-aware answers |
| Keyword fallback | No API key | Extracts matching lines from retrieved chunks |

---

### `core/document_processor.py`

Reads every `.md` file from `techcorp-docs/`, splits it into overlapping chunks, and calls `vector_engine.add_documents`.

```
chunk_text(text, size=500, overlap=100)

  text:  |<──────── 500 chars ────────>|
                         |<──────── 500 chars ────────>|
         |<── 400 ──>|<── 100 overlap ──>|

  Ensures no chunk exceeds the model's 256-token limit.
  Preserves sentence context across chunk boundaries.

DocumentProcessor.process_all_documents()
  for each category dir in techcorp-docs/:
    for each .md file:
      content = file.read_text()
      chunks  = chunk_text(content)          # ~7–13 chunks per file
      metadatas = [{file, category}] × len(chunks)
      vector_engine.add_documents(chunks, metadatas)
```

**Ingestion results (current knowledge base):**

| Category | Files | Chunks |
| --- | --- | --- |
| engineering | 2 | 24 |
| finance | 1 | 9 |
| hr | 4 | 35 |
| legal | 1 | 8 |
| products | 3 | 25 |
| **Total** | **11** | **101** |

---

### `app.py` — Flask application

```
Startup sequence
  1. init_telemetry()              # must run before VectorEngine
  2. VectorEngine()                # loads model + opens ChromaDB
  3. ChatEngine(vector_engine)     # checks for API key
  4. DocumentProcessor(vector_engine)
  5. if not vector_engine.is_initialized():
         doc_processor.process_all_documents()
  6. app.run(host='0.0.0.0', port=5252, debug=True)

Routes
  GET  /                  → render templates/chat.html
  POST /api/chat          → chat_engine.get_response(message)
  POST /api/chat/stream   → same, word-by-word SSE (50 ms token delay)
  GET  /api/status        → vector_engine.get_stats()
```

### `templates/chat.html` — Frontend

- Pure HTML / CSS / JavaScript — no build step, no npm, no framework
- Talks to `/api/chat` via `fetch` (JSON)
- Suggestion chips pre-fill the input with common questions
- Auto-resizing textarea; Enter to send, Shift+Enter for newline
- Shows sources and confidence score beneath each assistant reply
