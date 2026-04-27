# TechCorp AI Assistant — RAG Part I

A production-grade **Retrieval-Augmented Generation** (RAG) chat application implementing all concepts from Part I (Sections 1–5) of the RAG Comprehensive Guide. Employees ask natural-language questions; the system retrieves grounded evidence from a local knowledge base and generates accurate, cited answers.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Module Reference](#module-reference)
- [Project Structure](#project-structure)
- [Knowledge Base](#knowledge-base)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [RAG Architectures](#rag-architectures)
- [Retrieval Strategies](#retrieval-strategies)
- [Evaluation](#evaluation)
- [Telemetry](#telemetry)
- [Low-Level Design](#low-level-design)

---

## Architecture Overview

```
Browser / cURL
      │  HTTP
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Flask  app.py  — 8 REST endpoints                              │
│  /api/chat  /api/chat/stream  /api/ingest  /api/evaluate        │
│  /api/status  /api/architectures  /api/strategies               │
│  /api/conversation/reset                                        │
└──────┬────────────────┬───────────────────┬─────────────────────┘
       │                │                   │
       ▼                ▼                   ▼
┌────────────┐  ┌──────────────┐  ┌────────────────────┐
│ ChatEngine │  │  Retriever   │  │ DocumentProcessor  │
│            │  │              │  │                    │
│ Dispatches │  │ top_k        │  │ PDF / HTML / MD    │
│ to arch:   │  │ hybrid BM25+ │  │ code chunking      │
│ • simple   │  │ multi_query  │  │ fixed / recursive  │
│ • multihop │  │ hyde         │  │ rich metadata      │
│ • conv     │  │ filtered     │  │ content-hash IDs   │
│ • agentic  │  └──────┬───────┘  └─────────┬──────────┘
│ • crag     │         │                    │
│ • graph    │         ▼                    ▼
└────────────┘  ┌──────────────────────────────────────┐
                │          VectorEngine                │
                │                                      │
                │  EmbeddingBackend                    │
                │    • SentenceTransformers (default)  │
                │    • OpenAI Matryoshka               │
                │                                      │
                │  Storage                             │
                │    • ChromaDB (persistent, cosine)   │
                │    • FAISS (optional, in-memory)     │
                │                                      │
                │  BM25Index (rank_bm25)               │
                └──────────────────────────────────────┘
```

### Request Data Flow

```
User question
     │
     ▼
[1] Retrieve  ─── strategy ───►  chunks + similarity scores
     │             top_k / hybrid / multi_query / hyde / filtered
     ▼
[2] Architecture ─────────────►  SimpleRAG  /  MultiHop  /  Conv
     │                            Agentic   /  CRAG       /  Graph
     ▼
[3] Build prompt  ────────────►  RAG / CoT / few-shot / ReAct
     │
     ▼
[4] Generate  ────── provider ►  Gemini → Anthropic → OpenRouter → fallback
     │
     ▼
[5] Return  { answer, sources, confidence, architecture, strategy }
```

---

## Module Reference

| Module | Section | Responsibility |
|---|---|---|
| `core/foundations.py` | §1 | Tokenisation, similarity metrics, context budget, prompt templates, chunk IDs |
| `core/document_processor.py` | §2 | PDF/HTML/Markdown/code parsing, fixed + recursive chunking, metadata, traceability |
| `core/vector_engine.py` | §3 | Embedding backends, ChromaDB, FAISS index, batch encoding, score filtering |
| `core/retrieval.py` | §4 | BM25, RRF fusion, Retriever with 5 strategies, ACL/date/doctype filters |
| `core/rag_architectures.py` | §5 | SimpleRAG, MultiHop, Conversational, Agentic (ReAct), CRAG, Graph RAG |
| `core/evaluation.py` | §5 | Golden dataset, RAGEvaluator, BLEU-1, faithfulness, human eval checklist |
| `core/chat_engine.py` | §5 | LLM provider wiring, architecture dispatch, BM25 lifecycle |
| `core/telemetry.py` | — | OpenTelemetry TracerProvider (console or OTLP) |

---

## Project Structure

```
rag/
├── app.py                      # Flask server — 8 REST endpoints
├── Makefile                    # Common dev tasks
│
├── core/
│   ├── foundations.py          # §1  Tokenisation, similarity, prompts, chunk IDs
│   ├── document_processor.py   # §2  Multi-format parsing + chunking
│   ├── vector_engine.py        # §3  Embeddings, ChromaDB, FAISS
│   ├── retrieval.py            # §4  BM25, hybrid RRF, multi-query, HyDE, filters
│   ├── rag_architectures.py    # §5  Six RAG architecture classes
│   ├── evaluation.py           # §5  Golden dataset + RAGEvaluator
│   ├── chat_engine.py          # §5  LLM adapters + architecture dispatch
│   └── telemetry.py            # OTEL TracerProvider setup
│
├── templates/
│   └── chat.html               # Browser chat UI (vanilla JS, no framework)
│
├── techcorp-docs/              # Source knowledge base (Markdown)
│   ├── engineering/            #   on_call_runbook.md, tech_stack.md
│   ├── finance/                #   expense_policy.md
│   ├── hr/                     #   benefits.md, code_of_conduct.md, pet_policy.md, remote_work.md
│   ├── legal/                  #   data_privacy_policy.md
│   └── products/               #   cloudsync_pro.md, databridge.md, techassist_ai.md
│
├── chroma_db/                  # Persisted vector store (auto-created on first run)
├── eval_logs/                  # Evaluation run reports (JSONL)
│
├── test_chunking.py            # Chunking unit tests
├── test_embeddings.py          # Embedding similarity tests
├── test_search.py              # Semantic search smoke test
├── test_rag_pipeline.py        # End-to-end pipeline test
├── init_vectordb.py            # Create / reset ChromaDB collection
└── ingest_documents.py         # Manual re-ingest script
```

---

## Knowledge Base

11 Markdown documents across 5 categories, auto-ingested as **101 chunks** on first run.

| Category | File | Topics |
|---|---|---|
| `hr` | `benefits.md` | Medical/dental/vision, 401k, PTO, parental leave, learning budget |
| `hr` | `code_of_conduct.md` | Integrity, anti-harassment, conflict-of-interest, reporting |
| `hr` | `pet_policy.md` | Furry Fridays, eligible animals, designated areas, office mascot |
| `hr` | `remote_work.md` | Hybrid schedule, core hours, VPN, equipment stipends |
| `engineering` | `on_call_runbook.md` | Severity levels, escalation paths, incident playbooks, PIR |
| `engineering` | `tech_stack.md` | Languages, databases, Kubernetes, CI/CD, code review |
| `finance` | `expense_policy.md` | Travel limits, meal allowances, submission process |
| `legal` | `data_privacy_policy.md` | GDPR/CCPA, data retention, breach notification |
| `products` | `cloudsync_pro.md` | Features, pricing, SLA, system requirements |
| `products` | `databridge.md` | ETL connectors, pipeline modes, pricing |
| `products` | `techassist_ai.md` | AI support platform, integrations, onboarding |

---

## Quick Start

```bash
# 1. Clone and enter the project
cd /path/to/rag

# 2. Install dependencies (or use the Makefile)
make install

# 3. (Optional) set an LLM API key for fluent answers
export GEMINI_API_KEY="AIza..."       # highest priority
# OR
export ANTHROPIC_API_KEY="sk-ant-..."
# OR
export OPENROUTER_API_KEY="sk-or-..."

# 4. Run the application
make run
# → http://localhost:5252

# 5. Run evaluation on the golden dataset
make evaluate
```

---

## Configuration

### LLM Provider Priority

Provider is selected automatically: **Gemini → Anthropic → OpenRouter → keyword fallback**

| Provider | Env Var | Model |
|---|---|---|
| Gemini (default) | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` |
| OpenRouter | `OPENROUTER_API_KEY` + `OPENROUTER_MODEL` | `openai/gpt-4o-mini` (default) |
| Fallback | _(none required)_ | Keyword extraction from context |

### Architecture & Retrieval Defaults

Override at request time via JSON body, or change defaults in `app.py`:

| Setting | Default | Options |
|---|---|---|
| Architecture | `simple` | `simple`, `multihop`, `conversational`, `agentic`, `crag`, `graph` |
| Retrieval strategy | `top_k` | `top_k`, `hybrid`, `multi_query`, `hyde`, `filtered` |
| n_results | `5` | any int |
| score_threshold | `0.70` | 0.0 – 1.0 |

### Embedding Backend

Change in `app.py` → `VectorEngine(embedding_backend=..., embedding_model=...)`:

| Backend | Model alias | Dimension | Notes |
|---|---|---|---|
| `sentence_transformers` | `default` / `minilm` | 384 | Ships with the project, works offline |
| `sentence_transformers` | `bge` | 1024 | Higher quality, requires ~1 GB download |
| `openai` | `text-embedding-3-small` | 1536 | Needs `OPENAI_API_KEY`; supports Matryoshka |
| `openai` | `text-embedding-3-large` | 3072 | Best quality; Matryoshka via `dimensions=` |

### Chunking Strategy

Change in `app.py` → `DocumentProcessor(chunk_strategy=...)`:

| Strategy | Config key | Description |
|---|---|---|
| Recursive (default) | `recursive` | Tries `\n\n → \n → ". " → " " → ""` before hard split |
| Fixed character | `fixed_char` | Equal-size windows with `size=1000, overlap=100` |
| Fixed token | `fixed_token` | Token-aware via tiktoken; falls back to char/4 offline |

---

## API Reference

### `POST /api/chat`

Ask a question. Accepts an optional architecture and retrieval strategy.

**Request**
```json
{
  "message": "What is the hotel expense limit?",
  "architecture": "simple",
  "retrieval_strategy": "hybrid",
  "access_level": "public"
}
```

**Response**
```json
{
  "response": "The hotel expense limit for domestic travel is $250/night...",
  "sources": [
    { "category": "finance", "file": "expense_policy.md", "relevance": 84.2 }
  ],
  "confidence": 84.2,
  "provider": "gemini",
  "architecture": "simple",
  "retrieval_strategy": "hybrid",
  "metadata": { "n_chunks": 3 },
  "timestamp": "2026-04-23T12:00:00"
}
```

---

### `POST /api/chat/stream`

Same as `/api/chat` but streams the answer token-by-token via Server-Sent Events.

```
data: {"event": "start"}
data: {"event": "token", "content": "The "}
data: {"event": "token", "content": "hotel "}
...
data: {"event": "sources", "sources": [...], "confidence": 84.2}
data: {"event": "done"}
```

---

### `POST /api/ingest`

Add new content to the vector store at runtime.

**Request**
```json
{
  "type": "text",
  "content": "TechCorp now provides a $1,000 home office stipend annually.",
  "source": "hr/home_office_2026.md",
  "access_level": "internal"
}
```
`type` is one of `text` (plain), `html` (BeautifulSoup cleaned), or `pdf_path` (server-side file path).

**Response**
```json
{ "status": "success", "chunks_ingested": 2, "source": "hr/home_office_2026.md" }
```

---

### `POST /api/evaluate`

Run baseline evaluation over the built-in golden dataset.

**Request**
```json
{ "subset": 5, "architecture": "simple", "strategy": "top_k" }
```

**Response** — `EvalReport` dataclass serialised as JSON:
```json
{
  "run_id": "eval_1776947458",
  "n_questions": 5,
  "avg_retrieval_relevance": 0.73,
  "avg_faithfulness": 0.81,
  "exact_match_rate": 0.0,
  "avg_bleu1": 0.44,
  "avg_latency_ms": 320.5,
  "per_question": [...]
}
```

---

### `GET /api/status`

```json
{
  "status": "operational",
  "chunks": 101,
  "embedding_backend": "sentence_transformers",
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "faiss_index_size": 0,
  "provider": "gemini"
}
```

---

### `GET /api/architectures` / `GET /api/strategies`

Lists all available architectures / retrieval strategies with descriptions.

---

### `POST /api/conversation/reset`

Clears the conversational RAG chat history (memory window).

---

## RAG Architectures

### Simple RAG (§5.1)
Standard pipeline: retrieve → build prompt → generate. Supports chain-of-thought mode.

### Multi-Hop RAG (§5.2)
Iterative decomposition loop. The LLM emits JSON `{"action": "search", "query": "..."}` steps until it signals `"action": "done"`. Evidence accumulates across hops (up to 4 by default).

### Conversational RAG (§5.3)
Follow-up questions are rewritten into standalone questions via `build_query_contextualise_prompt`. A sliding window of the last 6 turns is included in the augmented context.

### Agentic RAG — ReAct (§5.4)
`Thought → Action[input] → Observation` loop. Built-in tools: `search(query)`, `summarise(text)`, `finish(answer)`. Custom tools injectable via `tools={"name": callable}`.

### CRAG — Corrective RAG (§5.5)
Every retrieved chunk is graded RELEVANT / NOT_RELEVANT by the LLM. If all chunks fail grading, a web-search fallback (or multi-query expansion) is triggered before generation.

### Graph RAG (§5.6)
LLM extracts `(subject | relation | object)` triples from ingested text into a NetworkX DiGraph. At query time, entities are identified and the ego-graph (radius=2) provides structured facts alongside retrieved passages.

---

## Retrieval Strategies

### Top-K (§4.1)
Dense cosine similarity search. Chunks with similarity < 0.70 are rejected.

### Hybrid BM25 + Vector with RRF (§4.2)
BM25 lexical scores and dense vector scores are merged using **Reciprocal Rank Fusion** (k=60):

```
RRF_score(d) = Σ  1 / (60 + rank_i(d))
```

### Multi-Query (§4.3)
The LLM generates 3 alternative phrasings of the query. Results across all queries are deduplicated by chunk ID and sorted by similarity.

### HyDE (§4.4)
A hypothetical answer is generated for the query and embedded. The hypothetical embedding — which lies in "answer space" — is used as the search vector instead of the raw query embedding.

### Filtered (§4.5)
Top-K search followed by post-filtering on:
- `doc_type` — `pdf`, `markdown`, `html`, `code`, `text`
- `access_level` — ACL hierarchy: `public < internal < confidential < restricted`
- `ingested_at` — ISO 8601 date range (`after` / `before`)

---

## Evaluation

### Golden Dataset
Five built-in Q&A pairs covering RAG concepts, chunking, hybrid search, HyDE, and score thresholds. Extend by calling `dataset.add(GoldenExample(...))` or saving a JSON file.

### Automated Metrics

| Metric | Implementation |
|---|---|
| Retrieval Relevance | Fraction of retrieved chunks with ≥ 2 keyword overlaps with the question |
| Faithfulness | Fraction of answer sentences grounded in retrieved context (word overlap proxy) |
| Exact Match | Lowercased string equality |
| BLEU-1 | Unigram precision of hypothesis vs. reference |
| Latency (ms) | End-to-end wall-clock time per question |

### Human Eval Checklist (§5.6)
Four binary dimensions per question:
1. **Retrieval relevance** — are the retrieved chunks relevant?
2. **Faithfulness** — does the answer stay within retrieved context?
3. **Completeness** — does it address all parts of the question?
4. **No hallucination** — does it avoid inventing facts not in context?

Run interactively:
```python
from core.evaluation import RAGEvaluator
evaluator.human_eval(results, interactive=True)
```

---

## Telemetry

### 1. ChromaDB product telemetry
Sends anonymous operation events (client version, OS, API calls — never document content) to ChromaDB's PostHog backend. Disable: `anonymized_telemetry=False` in `core/vector_engine.py`.

### 2. OpenTelemetry spans

| Span | Key attributes |
|---|---|
| `vector_engine.add_documents` | `chunk_count`, `category`, `embedding_dim`, `collection_size_after` |
| `vector_engine.encode` | `input_count`, `total_input_chars`, `embedding_dim`, `duration_ms` |
| `vector_engine.search` | `query`, `n_results_requested`, `n_results_returned`, `top_similarity_score`, `score_threshold` |
| `vector_engine.get_stats` | `total_chunks` |

**Switch to a production backend:**
```python
# app.py
init_telemetry(service_name="techcorp-rag", console=False)
```
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # Jaeger / Tempo / Honeycomb
```

---

## Low-Level Design

### Section 1 — Foundations (`core/foundations.py`)

```
count_tokens(text)
  ├── tiktoken BPE cl100k_base (lazy-loaded; falls back to len/4 offline)
  └── returns int

plan_context_budget(system, history, chunks, reserved=1024, model="default")
  ├── CONTEXT_LIMITS[model] = 32_000 / 128_000 / 200_000 / 1_048_576
  ├── system_tokens + history_tokens + chunk_tokens ≤ limit - reserved
  └── returns {chunks_included, chunk_indices_included, fits_all, ...}

cosine_similarity(a, b)  →  dot(a,b) / (|a| × |b|)
score_label(sim)         →  "very_high" ≥ 0.85 | "good" ≥ 0.70 | "moderate" ≥ 0.50 | "low"
make_chunk_id(src, idx, content) → sha256[:12] content hash — idempotent re-indexing
```

### Section 2 — Document Processor (`core/document_processor.py`)

```
DocumentProcessor._chunk(text)
  ├── "recursive"    → recursive_chunk(separators=["\n\n", "\n", ". ", " ", ""])
  ├── "fixed_char"   → fixed_char_chunk(size=1000, overlap=100)
  └── "fixed_token"  → fixed_token_chunk(max_tokens=256, overlap=32)

parse_markdown_by_headers(text) → [{section, level, content}, ...]   # split on H1/H2/H3
parse_html(html)                → BeautifulSoup4 noise removal + heading preservation
parse_pdf(path)                 → pdfplumber pages with table→markdown conversion
chunk_code(code, language)      → AST top-level defs (Python) or blank-line blocks

build_metadata(source, doc_type, chunk_index, total_chunks, ...)
  → {source, doc_type, chunk_index, total_chunks, access_level,
     ingested_at, page_number?, section?, file, category, token_count}
```

### Section 3 — Vector Engine (`core/vector_engine.py`)

```
EmbeddingBackend(backend, model_name)
  ├── "sentence_transformers" + "minilm"  → all-MiniLM-L6-v2  (384-dim)
  ├── "sentence_transformers" + "bge"     → BAAI/bge-large-en-v1.5  (1024-dim)
  └── "openai" + model_name               → text-embedding-3-small/large (Matryoshka)

FaissIndex(dim, index_type)
  ├── "flat"  → IndexFlatIP  (exact inner-product; L2-normalised → cosine)
  └── "hnsw"  → IndexHNSWFlat(M=32)  (ANN, faster for large corpora)

VectorEngine.search(query, n_results, score_threshold=0.70, where=None, use_faiss=False)
  ├── encode query   → 384-dim vector
  ├── ChromaDB cosine distance d  → similarity = 1 − d
  ├── filter: similarity ≥ score_threshold
  └── returns {documents, metadatas, distances, similarities}
```

### Section 4 — Retrieval (`core/retrieval.py`)

```
BM25Index.build(vector_engine)
  └── pulls all docs from ChromaDB.get() → BM25Okapi(tokenised_docs)

reciprocal_rank_fusion(ranked_lists, k=60)
  └── score(doc) = Σ 1/(60 + rank_i)  →  merged list sorted by score

Retriever.retrieve(query, strategy, n, user_access_level)
  ├── "top_k"       → VectorEngine.search + ACL filter
  ├── "hybrid"      → BM25.search + top_k  → RRF fusion
  ├── "multi_query" → LLM generates 3 variants → union by chunk_id
  ├── "hyde"        → LLM writes hypothetical doc → embed → search
  └── "filtered"    → top_k × 4 → doctype + date + ACL filters
```

### Section 5 — Architectures (`core/rag_architectures.py`)

```
SimpleRAG.run(question, strategy)
  └── retrieve → build_rag_prompt / build_chain_of_thought_prompt → llm_fn

MultiHopRAG.run(question, max_hops=4)
  └── for hop in range(max_hops):
        llm_fn(plan_prompt) → {action, query}  |  {action: "done", answer}
        if search: retrieve(sub_query) → accumulate evidence

ConversationalRAG.run(question)
  └── llm_fn(contextualise_prompt) → standalone_question
      retrieve(standalone) → llm_fn(history + context)
      update chat_history (sliding window of 6 turns)

AgenticRAG.run(question, max_steps=8)
  └── for step in range(max_steps):
        llm_fn(react_prompt) → Thought / Action[input] / finish
        tool_fn(action_input) → observation → append to trajectory

CRAG.run(question)
  └── retrieve → grade each chunk (RELEVANT / NOT_RELEVANT)
      → answer from relevant chunks
      → if none: web_search_fn or multi_query fallback

GraphRAG.run(question)
  └── retrieve seed chunks
      llm_fn(triple_prompt) → (subj | rel | obj) triples → NetworkX DiGraph
      llm_fn(entity_prompt) → entities → ego_graph(radius=2)
      answer from graph_context + retrieval_context
```
