"""
Section 4 – Core Retrieval Techniques

Implements:
  • Top-K similarity search with score-threshold filtering (wraps VectorEngine)
  • Hybrid search: BM25 (rank_bm25) + dense vector search, fused with
    Reciprocal Rank Fusion (RRF, k=60)
  • Metadata filtering — date-range, access-level (ACL), doc-type
  • Multi-query retrieval — LLM generates N query variants, deduplicates by chunk ID
  • HyDE (Hypothetical Document Embeddings) — embed hypothetical answer, search
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.foundations import (
    SCORE_REJECT_THRESHOLD,
    build_query_expansion_prompt,
    build_hyde_prompt,
)

# ---------------------------------------------------------------------------
# BM25 back-end
# ---------------------------------------------------------------------------

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def _tokenise(text: str) -> List[str]:
    """Simple whitespace + lower-case tokeniser for BM25."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# BM25 Index — built lazily over the corpus stored in VectorEngine
# ---------------------------------------------------------------------------

class BM25Index:
    """
    BM25 index built over the documents already stored in the ChromaDB collection.
    Call `build()` after documents are ingested to (re-)index them.
    """

    def __init__(self) -> None:
        self._bm25: Optional[Any] = None
        self._docs: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._ids: List[str] = []

    def build(self, vector_engine) -> None:
        """
        Pull every document from the ChromaDB collection and build a BM25 index.
        ChromaDB's get() returns all stored chunks.
        """
        if not BM25_AVAILABLE:
            print("[BM25Index] rank_bm25 not installed — BM25 unavailable")
            return

        result = vector_engine.collection.get(include=["documents", "metadatas"])
        self._docs = result.get("documents", [])
        self._metas = result.get("metadatas", [])
        self._ids = result.get("ids", [])

        if not self._docs:
            print("[BM25Index] No documents found — BM25 index empty")
            return

        tokenised = [_tokenise(d) for d in self._docs]
        self._bm25 = BM25Okapi(tokenised)
        print(f"[BM25Index] Built over {len(self._docs)} documents")

    def search(
        self,
        query: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return top-n BM25 hits as list of {"id", "text", "metadata", "bm25_score"}.
        """
        if self._bm25 is None or not self._docs:
            return []

        q_tokens = _tokenise(query)
        scores = self._bm25.get_scores(q_tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n_results]

        return [
            {
                "id": self._ids[i],
                "text": self._docs[i],
                "metadata": self._metas[i],
                "bm25_score": float(score),
            }
            for i, score in ranked
            if score > 0
        ]

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self._docs)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    id_key: str = "id",
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ  1 / (k + rank_i)   for each list that contains the item.

    k=60 is the standard value from the original RRF paper (Cormack et al., 2009).
    Higher k makes the fusion less sensitive to absolute rank differences.
    """
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[id_key]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in payloads:
                payloads[doc_id] = item

    merged = sorted(payloads.values(), key=lambda x: scores[x[id_key]], reverse=True)

    # Attach the RRF score for transparency
    for item in merged:
        item["rrf_score"] = round(scores[item[id_key]], 6)

    return merged


# ---------------------------------------------------------------------------
# Metadata Filters
# ---------------------------------------------------------------------------

def apply_date_filter(
    results: List[Dict[str, Any]],
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter chunks by ingested_at timestamp (ISO 8601).
    after / before are inclusive bounds (also ISO 8601 strings).
    """
    filtered = []
    for item in results:
        ts = item.get("metadata", {}).get("ingested_at", "")
        if after and ts < after:
            continue
        if before and ts > before:
            continue
        filtered.append(item)
    return filtered


def apply_acl_filter(
    results: List[Dict[str, Any]],
    user_access_level: str = "public",
) -> List[Dict[str, Any]]:
    """
    Access-Control Layer (ACL) filter.
    access_level hierarchy: public < internal < confidential < restricted.
    A user with level X can see all levels ≤ X.
    """
    _LEVELS = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
    user_rank = _LEVELS.get(user_access_level, 0)
    return [
        item for item in results
        if _LEVELS.get(item.get("metadata", {}).get("access_level", "public"), 0) <= user_rank
    ]


def apply_doctype_filter(
    results: List[Dict[str, Any]],
    doc_types: List[str],
) -> List[Dict[str, Any]]:
    """Keep only chunks whose doc_type is in doc_types."""
    if not doc_types:
        return results
    return [
        item for item in results
        if item.get("metadata", {}).get("doc_type", "") in doc_types
    ]


# ---------------------------------------------------------------------------
# Retriever — high-level interface
# ---------------------------------------------------------------------------

class Retriever:
    """
    Encapsulates all Section 4 retrieval strategies.
    Requires a VectorEngine instance; optionally accepts a BM25Index and an LLM
    callable (any function that takes a prompt str and returns str).
    """

    def __init__(
        self,
        vector_engine,
        bm25_index: Optional[BM25Index] = None,
        llm_fn: Optional[Any] = None,
        default_n: int = 5,
        score_threshold: float = SCORE_REJECT_THRESHOLD,
    ) -> None:
        self.vector_engine = vector_engine
        self.bm25 = bm25_index
        self.llm_fn = llm_fn
        self.default_n = default_n
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Strategy 1: Top-K similarity search
    # ------------------------------------------------------------------

    def top_k_search(
        self,
        query: str,
        n: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        user_access_level: str = "public",
    ) -> List[Dict[str, Any]]:
        """
        Standard top-K vector similarity search with:
        • score-threshold filtering (drops similarity < 0.70)
        • ACL filtering
        """
        n = n or self.default_n
        results = self.vector_engine.search(
            query, n_results=n, score_threshold=self.score_threshold, where=where
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        sims = results["similarities"][0]
        ids_raw = []

        # ChromaDB doesn't return IDs in query results — reconstruct from metadata
        items = [
            {
                "id": metas[i].get("source", f"chunk_{i}") + f"__{i}",
                "text": docs[i],
                "metadata": metas[i],
                "similarity": sims[i],
            }
            for i in range(len(docs))
        ]
        return apply_acl_filter(items, user_access_level)

    # ------------------------------------------------------------------
    # Strategy 2: Hybrid BM25 + Vector with RRF
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        n: Optional[int] = None,
        rrf_k: int = 60,
        user_access_level: str = "public",
    ) -> List[Dict[str, Any]]:
        """
        Combine BM25 lexical retrieval and dense vector search via RRF.
        Falls back to top-K if BM25 index is not built.
        """
        n = n or self.default_n
        if self.bm25 is None or not self.bm25.is_ready:
            print("[Retriever] BM25 index not ready — falling back to top-K")
            return self.top_k_search(query, n=n, user_access_level=user_access_level)

        # BM25 hits
        bm25_hits = self.bm25.search(query, n_results=n * 2)

        # Dense hits
        dense_hits = self.top_k_search(query, n=n * 2, user_access_level=user_access_level)

        # Normalise dense hit IDs to match BM25 IDs
        for hit in dense_hits:
            meta = hit.get("metadata", {})
            hit["id"] = meta.get("source", "") + f"__c{meta.get('chunk_index', 0):04d}"

        fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=rrf_k)
        acl_filtered = apply_acl_filter(fused, user_access_level)
        return acl_filtered[:n]

    # ------------------------------------------------------------------
    # Strategy 3: Metadata filtering (date, ACL, doc-type)
    # ------------------------------------------------------------------

    def filtered_search(
        self,
        query: str,
        n: Optional[int] = None,
        doc_types: Optional[List[str]] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        user_access_level: str = "public",
    ) -> List[Dict[str, Any]]:
        """
        Top-K search + metadata post-filtering for date range, ACL, doc-type.
        Fetches 4× as many results before filtering to ensure n survive.
        """
        n = n or self.default_n
        raw = self.top_k_search(query, n=n * 4, user_access_level=user_access_level)
        if doc_types:
            raw = apply_doctype_filter(raw, doc_types)
        if after or before:
            raw = apply_date_filter(raw, after=after, before=before)
        return raw[:n]

    # ------------------------------------------------------------------
    # Strategy 4: Multi-query retrieval
    # ------------------------------------------------------------------

    def multi_query_search(
        self,
        query: str,
        n_variants: int = 3,
        n_per_query: int = 5,
        user_access_level: str = "public",
    ) -> List[Dict[str, Any]]:
        """
        Generate n_variants alternative phrasings of the query using the LLM,
        retrieve for each, then deduplicate by chunk ID (union of result sets).
        """
        if self.llm_fn is None:
            return self.top_k_search(query, n=n_per_query, user_access_level=user_access_level)

        # Generate query variants
        prompt = build_query_expansion_prompt(query, n=n_variants)
        try:
            variants_text = self.llm_fn(prompt)
            variants = [
                line.strip() for line in variants_text.strip().split("\n")
                if line.strip() and line.strip() != query
            ][:n_variants]
        except Exception as e:
            print(f"[multi_query] LLM call failed: {e}")
            variants = []

        all_queries = [query] + variants
        print(f"[multi_query] Queries: {all_queries}")

        seen_ids: set = set()
        merged: List[Dict[str, Any]] = []

        for q in all_queries:
            hits = self.top_k_search(q, n=n_per_query, user_access_level=user_access_level)
            for hit in hits:
                cid = hit.get("id") or hit.get("text", "")[:40]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged.append(hit)

        # Sort by similarity (best first) and cap
        merged.sort(key=lambda x: x.get("similarity", x.get("rrf_score", 0)), reverse=True)
        return merged[: n_per_query]

    # ------------------------------------------------------------------
    # Strategy 5: HyDE — Hypothetical Document Embeddings
    # ------------------------------------------------------------------

    def hyde_search(
        self,
        query: str,
        n: Optional[int] = None,
        user_access_level: str = "public",
    ) -> List[Dict[str, Any]]:
        """
        1. Ask the LLM to write a hypothetical answer to the query.
        2. Embed that hypothetical text.
        3. Use the hypothetical embedding as the query vector.

        This often improves retrieval for abstract or vague questions because the
        hypothetical document occupies a more "answer-like" region of embedding space.
        """
        n = n or self.default_n
        if self.llm_fn is None:
            return self.top_k_search(query, n=n, user_access_level=user_access_level)

        prompt = build_hyde_prompt(query)
        try:
            hypothetical_doc = self.llm_fn(prompt)
        except Exception as e:
            print(f"[HyDE] LLM call failed: {e} — falling back to standard search")
            return self.top_k_search(query, n=n, user_access_level=user_access_level)

        print(f"[HyDE] Hypothetical document ({len(hypothetical_doc)} chars) generated")

        # Search using the hypothetical doc as the query
        results = self.vector_engine.search(
            hypothetical_doc,
            n_results=n,
            score_threshold=self.score_threshold,
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        sims = results["similarities"][0]

        items = [
            {
                "id": metas[i].get("source", "") + f"__{i}",
                "text": docs[i],
                "metadata": metas[i],
                "similarity": sims[i],
                "hyde_used": True,
            }
            for i in range(len(docs))
        ]
        return apply_acl_filter(items, user_access_level)

    # ------------------------------------------------------------------
    # Unified dispatch
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        strategy: str = "top_k",
        n: Optional[int] = None,
        user_access_level: str = "public",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Dispatch to a retrieval strategy by name.
        strategy: "top_k" | "hybrid" | "filtered" | "multi_query" | "hyde"
        Extra kwargs are forwarded to the chosen strategy.
        """
        n = n or self.default_n
        if strategy == "hybrid":
            return self.hybrid_search(query, n=n, user_access_level=user_access_level, **kwargs)
        if strategy == "filtered":
            return self.filtered_search(query, n=n, user_access_level=user_access_level, **kwargs)
        if strategy == "multi_query":
            return self.multi_query_search(query, n_per_query=n, user_access_level=user_access_level, **kwargs)
        if strategy == "hyde":
            return self.hyde_search(query, n=n, user_access_level=user_access_level, **kwargs)
        return self.top_k_search(query, n=n, user_access_level=user_access_level, **kwargs)
