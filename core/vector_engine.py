"""
Vector Engine - manages ChromaDB storage and semantic search.

Telemetry layers
----------------
1. ChromaDB product telemetry (anonymized_telemetry=True):
   - What it captures: client version, Python version, OS platform, and the
     names of API operations executed (collection.add, collection.query, …).
     No document content or query text is ever included.
   - Where it goes: Chroma's PostHog analytics instance (analytics.chroma.com).
   - Purpose: helps the ChromaDB team understand feature adoption and usage
     patterns so they can prioritise development.
   - Note: in ChromaDB >=1.0 the PostHog backend is a no-op stub; the flag is
     kept for forward compatibility and for deployments against a Chroma server
     that still honours it.

2. OpenTelemetry (OTEL) spans — emitted by this class around every operation:
   - Span: vector_engine.add_documents
       attributes: chunk_count, category, embedding_dim
   - Span: vector_engine.encode
       attributes: input_length, embedding_dim, duration_ms
   - Span: vector_engine.search
       attributes: query (first 120 chars), n_results_requested,
                   n_results_returned, top_similarity_score, collection_size
   - Span: vector_engine.get_stats
       attributes: total_chunks
"""

import time

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime
from opentelemetry import trace

from core.telemetry import get_tracer


class VectorEngine:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "techcorp_docs"):
        self.db_path = db_path
        self.collection_name = collection_name
        self._tracer: trace.Tracer = get_tracer()

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # anonymized_telemetry=True (default):
        #   ChromaDB sends anonymised operation events to its PostHog backend.
        #   Events contain: client version, Python version, OS, operation names.
        #   No document content, embeddings, or query text is included.
        #   Set to False to opt out entirely.
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=True,          # re-enabled — see docstring
                chroma_otel_service_name="techcorp-rag",
                chroma_otel_granularity="operation", # emit a span per operation
            ),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._last_updated: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        return self.collection.count() > 0

    def add_documents(self, chunks: list, metadatas: list | None = None) -> None:
        if not chunks:
            return

        category = (metadatas[0].get("category", "unknown") if metadatas else "unknown")

        with self._tracer.start_as_current_span("vector_engine.add_documents") as span:
            span.set_attribute("chunk_count", len(chunks))
            span.set_attribute("category", category)

            embeddings = self._encode(chunks)
            span.set_attribute("embedding_dim", len(embeddings[0]) if embeddings else 0)

            ids = [f"doc_{self.collection.count() + i}" for i in range(len(chunks))]
            if metadatas is None:
                metadatas = [{"source": "unknown"}] * len(chunks)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
            self._last_updated = datetime.now().isoformat()
            span.set_attribute("collection_size_after", self.collection.count())

    def search(self, query: str, n_results: int = 3) -> dict:
        if self.collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        with self._tracer.start_as_current_span("vector_engine.search") as span:
            span.set_attribute("query", query[:120])          # avoid huge attrs
            span.set_attribute("n_results_requested", n_results)
            span.set_attribute("collection_size", self.collection.count())

            query_embedding = self._encode([query])[0]

            actual_n = min(n_results, self.collection.count())
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_n,
            )

            n_returned = len(results["documents"][0])
            span.set_attribute("n_results_returned", n_returned)

            distances = results.get("distances", [[]])[0]
            if distances:
                top_sim = round(1 - distances[0], 4)
                span.set_attribute("top_similarity_score", top_sim)

            return results

    def get_stats(self) -> dict:
        with self._tracer.start_as_current_span("vector_engine.get_stats") as span:
            count = self.collection.count()
            span.set_attribute("total_chunks", count)
            return {
                "total_documents": 1,
                "total_chunks": count,
                "last_updated": self._last_updated or "N/A",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list) -> list:
        with self._tracer.start_as_current_span("vector_engine.encode") as span:
            t0 = time.perf_counter()
            span.set_attribute("input_count", len(texts))
            span.set_attribute("total_input_chars", sum(len(t) for t in texts))

            embeddings = self.model.encode(texts).tolist()

            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            span.set_attribute("embedding_dim", len(embeddings[0]) if embeddings else 0)
            span.set_attribute("duration_ms", duration_ms)
        return embeddings
