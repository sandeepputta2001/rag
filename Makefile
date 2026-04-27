## TechCorp RAG — Part I
## Usage: make <target>

PYTHON     := venv/bin/python
PIP        := venv/bin/python -m pip
PORT       := 5252
TMPDIR     := /home/sandeep/pip_tmp

export TMPDIR

.PHONY: help install install-dev run dev reset ingest evaluate \
        test test-chunking test-embeddings test-search test-pipeline \
        status clean-cache clean

# ── Default ──────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  TechCorp AI Assistant — RAG Part I"
	@echo ""
	@echo "  Core targets"
	@echo "    make install        Install all Python dependencies into venv"
	@echo "    make run            Start the Flask app on port $(PORT)"
	@echo "    make dev            Start with FLASK_DEBUG=1 and OTEL console output"
	@echo ""
	@echo "  Data targets"
	@echo "    make ingest         Re-ingest all techcorp-docs into ChromaDB"
	@echo "    make reset          Wipe chroma_db/ and re-ingest from scratch"
	@echo ""
	@echo "  Evaluation"
	@echo "    make evaluate       Run evaluation on all 5 golden-dataset questions"
	@echo "    make evaluate-fast  Run evaluation on first 3 questions only"
	@echo ""
	@echo "  Tests"
	@echo "    make test           Run all test scripts"
	@echo "    make test-chunking  Test recursive + fixed chunking"
	@echo "    make test-embeddings Test sentence-transformer similarity"
	@echo "    make test-search    Semantic search smoke test"
	@echo "    make test-pipeline  End-to-end RAG pipeline smoke test"
	@echo ""
	@echo "  Utilities"
	@echo "    make status         Print vector store stats via /api/status"
	@echo "    make clean-cache    Remove Python __pycache__ directories"
	@echo "    make clean          clean-cache + remove eval_logs/"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────

install:
	@echo "[install] Installing dependencies..."
	$(PIP) install --no-cache-dir \
		flask \
		chromadb \
		sentence-transformers \
		opentelemetry-sdk \
		opentelemetry-api \
		opentelemetry-exporter-otlp-proto-grpc \
		pdfplumber \
		beautifulsoup4 \
		rank-bm25 \
		faiss-cpu \
		tiktoken \
		networkx \
		python-dotenv
	@echo "[install] Done."

install-dev: install
	@echo "[install-dev] Installing optional dev dependencies..."
	$(PIP) install --no-cache-dir pytest httpie
	@echo "[install-dev] Done."

# ── Run ───────────────────────────────────────────────────────────────────────

run:
	@echo "[run] Starting TechCorp AI Assistant on http://localhost:$(PORT) ..."
	$(PYTHON) app.py

dev:
	@echo "[dev] Starting in debug mode with console telemetry..."
	FLASK_DEBUG=1 OTEL_CONSOLE=1 $(PYTHON) app.py

# ── Data ──────────────────────────────────────────────────────────────────────

ingest:
	@echo "[ingest] Re-ingesting techcorp-docs into ChromaDB..."
	$(PYTHON) -c "
from core.telemetry import init_telemetry
init_telemetry(console=False)
from core.vector_engine import VectorEngine
from core.document_processor import DocumentProcessor
ve = VectorEngine()
dp = DocumentProcessor(ve, chunk_strategy='recursive')
result = dp.process_all_documents()
print(f'Ingested: {result[\"documents\"]} docs, {result[\"chunks\"]} chunks')
"

reset:
	@echo "[reset] Wiping chroma_db/ and re-ingesting..."
	rm -rf chroma_db/
	$(MAKE) ingest

# ── Evaluation ────────────────────────────────────────────────────────────────

evaluate:
	@echo "[evaluate] Running baseline evaluation (5 questions)..."
	$(PYTHON) -c "
import sys; sys.path.insert(0, '.')
from core.telemetry import init_telemetry
init_telemetry(console=False)
from core.vector_engine import VectorEngine
from core.chat_engine import ChatEngine
from core.evaluation import GoldenDataset, RAGEvaluator

ve = VectorEngine()
ce = ChatEngine(ve)

def rag_fn(q):
    return ce.get_response(q)

ds = GoldenDataset().load()
ev = RAGEvaluator(rag_fn=rag_fn)
report = ev.evaluate(ds, architecture='simple', retrieval_strategy='top_k')
ev.print_report(report)
"

evaluate-fast:
	@echo "[evaluate-fast] Running evaluation on first 3 questions..."
	$(PYTHON) -c "
import sys; sys.path.insert(0, '.')
from core.telemetry import init_telemetry
init_telemetry(console=False)
from core.vector_engine import VectorEngine
from core.chat_engine import ChatEngine
from core.evaluation import GoldenDataset, RAGEvaluator

ve = VectorEngine()
ce = ChatEngine(ve)

def rag_fn(q):
    return ce.get_response(q)

ds = GoldenDataset().load()
ev = RAGEvaluator(rag_fn=rag_fn)
report = ev.evaluate(ds, subset=3)
ev.print_report(report)
"

# ── Tests ─────────────────────────────────────────────────────────────────────

test: test-chunking test-embeddings test-search test-pipeline
	@echo ""
	@echo "[test] All tests complete."

test-chunking:
	@echo "[test-chunking] Running chunking tests..."
	$(PYTHON) test_chunking.py

test-embeddings:
	@echo "[test-embeddings] Running embedding tests..."
	$(PYTHON) test_embeddings.py

test-search:
	@echo "[test-search] Running search smoke test..."
	$(PYTHON) test_search.py

test-pipeline:
	@echo "[test-pipeline] Running end-to-end pipeline test..."
	$(PYTHON) test_rag_pipeline.py

# ── Utilities ─────────────────────────────────────────────────────────────────

status:
	@echo "[status] Querying /api/status (app must be running on port $(PORT))..."
	curl -s http://localhost:$(PORT)/api/status | python3 -m json.tool

clean-cache:
	@echo "[clean-cache] Removing __pycache__ directories..."
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./venv/*" -delete 2>/dev/null || true
	@echo "[clean-cache] Done."

clean: clean-cache
	@echo "[clean] Removing eval_logs/..."
	rm -rf eval_logs/
	@echo "[clean] Done."
