"""
Section 3 – Standard Embeddings & Vector Databases

Implements:
  • Multiple embedding back-ends
      – Sentence Transformers (all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5)
      – OpenAI text-embedding-3-small / text-embedding-3-large (Matryoshka
        dimension reduction via the `dimensions` parameter)
  • FAISS-based vector index
      – IndexFlatIP  (exact cosine — vectors are L2-normalised before add/search)
      – IndexHNSWFlat (approximate nearest-neighbour for large corpora)
  • ChromaDB persistent storage (original back-end — kept for compatibility)
  • Batch embedding with order preservation
  • Score-threshold filtering (rejects chunks with similarity < 0.70)
  • OpenTelemetry spans around every operation

Telemetry layers
----------------
1. ChromaDB product telemetry (anonymized_telemetry=True) — same as before.
2. OTEL spans — emitted around add_documents, encode, search, get_stats.
"""

from __future__ import annotations

import os
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime
from opentelemetry import trace

from core.telemetry import get_tracer
from core.foundations import SCORE_REJECT_THRESHOLD, l2_normalize, cosine_similarity

# ---------------------------------------------------------------------------
# FAISS — optional; gracefully absent if not installed
# ---------------------------------------------------------------------------

try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Embedding back-end registry
# ---------------------------------------------------------------------------

SENTENCE_TRANSFORMER_MODELS = {
    "minilm":   "all-MiniLM-L6-v2",       # fast, 384-dim
    "bge":      "BAAI/bge-large-en-v1.5", # higher quality, 1024-dim
    "default":  "all-MiniLM-L6-v2",
}


class EmbeddingBackend:
    """
    Unified embedding interface supporting Sentence Transformers and OpenAI.
    """

    def __init__(self, backend: str = "sentence_transformers", model_name: str = "default") -> None:
        self.backend = backend
        self.dim: int = 0

        if backend == "sentence_transformers":
            resolved = SENTENCE_TRANSFORMER_MODELS.get(model_name, model_name)
            self.model = SentenceTransformer(resolved)
            get_dim = getattr(self.model, "get_embedding_dimension",
                              getattr(self.model, "get_sentence_embedding_dimension", None))
            self.dim = get_dim() if get_dim else 384
            self.model_name = resolved

        elif backend == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set — cannot use OpenAI embeddings")
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name if model_name not in ("default", "minilm", "bge") \
                else "text-embedding-3-small"
            # Matryoshka: reduce dimensions for text-embedding-3 series
            self.matryoshka_dim: Optional[int] = None
            self.dim = 1536  # text-embedding-3-small default

        else:
            raise ValueError(f"Unknown embedding backend: {backend!r}")

    def encode(self, texts: List[str], matryoshka_dim: Optional[int] = None) -> List[List[float]]:
        """
        Encode a list of texts into dense vectors.
        matryoshka_dim: truncate OpenAI Matryoshka embeddings to this dimension.
        """
        if self.backend == "sentence_transformers":
            vecs = self.model.encode(texts, normalize_embeddings=True).tolist()
            return vecs

        if self.backend == "openai":
            kwargs: Dict[str, Any] = {"input": texts, "model": self.model_name}
            dim = matryoshka_dim or self.matryoshka_dim
            if dim and "text-embedding-3" in self.model_name:
                kwargs["dimensions"] = dim
            resp = self.client.embeddings.create(**kwargs)
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]

        return []


# ---------------------------------------------------------------------------
# FAISS Index wrapper (Section 3.3)
# ---------------------------------------------------------------------------

class FaissIndex:
    """
    In-memory FAISS index with metadata storage.
    Supports exact inner-product search (IndexFlatIP) and approximate
    HNSW (IndexHNSWFlat) for large corpora.
    """

    def __init__(self, dim: int, index_type: str = "flat") -> None:
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is not installed. Run: pip install faiss-cpu")
        self.dim = dim
        if index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32)  # M=32
        else:
            self.index = faiss.IndexFlatIP(dim)  # exact, L2-norm'd → cosine
        self._docs: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._ids: List[str] = []

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """Add documents to the FAISS index. Vectors are L2-normalised."""
        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self._docs.extend(texts)
        self._metas.extend(metadatas)
        self._ids.extend(ids)

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        score_threshold: float = SCORE_REJECT_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Return top-n results with cosine similarity ≥ score_threshold.
        similarity = inner-product after L2 normalisation.
        """
        if self.index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)
        k = min(n_results, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sim = float(score)
            if sim < score_threshold:
                continue
            results.append({
                "id": self._ids[idx],
                "text": self._docs[idx],
                "metadata": self._metas[idx],
                "similarity": round(sim, 4),
            })
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal


# ---------------------------------------------------------------------------
# VectorEngine
# ---------------------------------------------------------------------------

class VectorEngine:
    """
    Unified vector store backed by ChromaDB (persistent) with an optional
    in-memory FAISS index for advanced retrieval (hybrid search, etc.).

    Embedding back-end is configurable; Sentence Transformers is the default.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "techcorp_docs",
        embedding_backend: str = "sentence_transformers",
        embedding_model: str = "default",
        use_faiss: bool = False,
        faiss_index_type: str = "flat",
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self._tracer: trace.Tracer = get_tracer()

        # --- Embedding back-end ---
        self.embedder = EmbeddingBackend(
            backend=embedding_backend,
            model_name=embedding_model,
        )
        print(f"[VectorEngine] Embedding: {self.embedder.backend}/{self.embedder.model_name} "
              f"(dim={self.embedder.dim})")

        # --- ChromaDB ---
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=True,
                chroma_otel_service_name="techcorp-rag",
                chroma_otel_granularity="operation",
            ),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # --- FAISS (optional) ---
        self._faiss: Optional[FaissIndex] = None
        if use_faiss:
            if FAISS_AVAILABLE:
                self._faiss = FaissIndex(self.embedder.dim, index_type=faiss_index_type)
                print(f"[VectorEngine] FAISS index: {faiss_index_type}")
            else:
                print("[VectorEngine] FAISS not installed — using ChromaDB only")

        self._last_updated: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        return self.collection.count() > 0

    def add_documents(
        self,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Embed and store chunks in ChromaDB (and FAISS if enabled).
        ids: optional list of pre-computed chunk IDs (content-hash style).
             If omitted, sequential IDs are generated.
        """
        if not chunks:
            return

        category = (metadatas[0].get("category", "unknown") if metadatas else "unknown")

        with self._tracer.start_as_current_span("vector_engine.add_documents") as span:
            span.set_attribute("chunk_count", len(chunks))
            span.set_attribute("category", category)

            embeddings = self._encode(chunks)
            span.set_attribute("embedding_dim", len(embeddings[0]) if embeddings else 0)

            # Fallback IDs — sequential, unique across collection
            if ids is None:
                base = self.collection.count()
                ids = [f"doc_{base + i}" for i in range(len(chunks))]

            if metadatas is None:
                metadatas = [{"source": "unknown"}] * len(chunks)

            # Ensure ChromaDB metadata values are primitive types
            safe_metas = [_sanitise_metadata(m) for m in metadatas]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=safe_metas,
            )

            # Mirror into FAISS
            if self._faiss is not None:
                self._faiss.add(chunks, embeddings, safe_metas, ids)

            self._last_updated = datetime.now().isoformat()
            span.set_attribute("collection_size_after", self.collection.count())

    def search(
        self,
        query: str,
        n_results: int = 5,
        score_threshold: float = SCORE_REJECT_THRESHOLD,
        where: Optional[Dict[str, Any]] = None,
        use_faiss: bool = False,
    ) -> Dict[str, Any]:
        """
        Semantic similarity search.

        where: ChromaDB metadata filter dict, e.g. {"doc_type": "pdf"}.
        use_faiss: route through FAISS index instead of ChromaDB.
        Returns dict with "documents", "metadatas", "distances", "similarities".
        """
        if self.collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "similarities": [[]]}

        with self._tracer.start_as_current_span("vector_engine.search") as span:
            span.set_attribute("query", query[:120])
            span.set_attribute("n_results_requested", n_results)
            span.set_attribute("score_threshold", score_threshold)
            span.set_attribute("collection_size", self.collection.count())

            q_embedding = self._encode([query])[0]

            if use_faiss and self._faiss is not None:
                return self._faiss_search(q_embedding, n_results, score_threshold, span)

            return self._chroma_search(q_embedding, n_results, score_threshold, where, span)

    def get_stats(self) -> Dict[str, Any]:
        with self._tracer.start_as_current_span("vector_engine.get_stats") as span:
            count = self.collection.count()
            faiss_count = self._faiss.size if self._faiss else 0
            span.set_attribute("total_chunks", count)
            return {
                "total_documents": 1,
                "total_chunks": count,
                "faiss_index_size": faiss_count,
                "embedding_backend": self.embedder.backend,
                "embedding_model": self.embedder.model_name,
                "embedding_dim": self.embedder.dim,
                "last_updated": self._last_updated or "N/A",
            }

    def get_embeddings_for_texts(self, texts: List[str]) -> List[List[float]]:
        """Public embedding access for retrieval modules (BM25 hybrid, HyDE, etc.)."""
        return self._encode(texts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> List[List[float]]:
        with self._tracer.start_as_current_span("vector_engine.encode") as span:
            t0 = time.perf_counter()
            span.set_attribute("input_count", len(texts))
            span.set_attribute("total_input_chars", sum(len(t) for t in texts))

            # Batch in groups of 128 to stay within model memory limits
            all_embeddings: List[List[float]] = []
            batch_size = 128
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                all_embeddings.extend(self.embedder.encode(batch))

            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            span.set_attribute("embedding_dim", len(all_embeddings[0]) if all_embeddings else 0)
            span.set_attribute("duration_ms", duration_ms)
        return all_embeddings

    def _chroma_search(
        self,
        q_embedding: List[float],
        n_results: int,
        score_threshold: float,
        where: Optional[Dict[str, Any]],
        span: Any,
    ) -> Dict[str, Any]:
        actual_n = min(n_results, self.collection.count())
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [q_embedding],
            "n_results": actual_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        raw_docs = results["documents"][0]
        raw_metas = results["metadatas"][0]
        raw_dists = results.get("distances", [[]])[0]

        # Convert ChromaDB cosine distance → similarity and filter
        filtered_docs, filtered_metas, filtered_dists, filtered_sims = [], [], [], []
        for doc, meta, dist in zip(raw_docs, raw_metas, raw_dists):
            sim = round(1.0 - dist, 4)
            if sim >= score_threshold:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)
                filtered_sims.append(sim)

        if filtered_sims:
            span.set_attribute("top_similarity_score", filtered_sims[0])
        span.set_attribute("n_results_returned", len(filtered_docs))

        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_metas],
            "distances": [filtered_dists],
            "similarities": [filtered_sims],
        }

    def _faiss_search(
        self,
        q_embedding: List[float],
        n_results: int,
        score_threshold: float,
        span: Any,
    ) -> Dict[str, Any]:
        hits = self._faiss.search(q_embedding, n_results, score_threshold)
        docs = [h["text"] for h in hits]
        metas = [h["metadata"] for h in hits]
        sims = [h["similarity"] for h in hits]
        dists = [round(1.0 - s, 4) for s in sims]

        if sims:
            span.set_attribute("top_similarity_score", sims[0])
        span.set_attribute("n_results_returned", len(docs))

        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
            "similarities": [sims],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB requires metadata values to be str, int, float, or bool.
    Convert everything else to str.
    """
    safe: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif v is None:
            safe[k] = ""
        else:
            safe[k] = str(v)
    return safe
