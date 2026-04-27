"""
Chat Engine — generates answers using retrieved context (RAG)

Provider priority (first available wins):
  1. Gemini      — set GEMINI_API_KEY
  2. Anthropic   — set ANTHROPIC_API_KEY
  3. OpenRouter  — set OPENROUTER_API_KEY (optionally OPENROUTER_MODEL)
  4. Keyword extraction fallback (no API key required)

Architecture modes (Section 5):
  • simple      — standard top-K retrieval + prompt + generate
  • multihop    — iterative sub-question decomposition loop
  • conversational — query contextualisation + chat memory
  • agentic     — ReAct tool-use loop
  • crag        — self-correcting with relevance grading
  • graph       — knowledge-graph enriched retrieval

Retrieval strategies (Section 4):
  • top_k       — vector similarity + score threshold
  • hybrid      — BM25 + vector with RRF
  • multi_query — LLM-generated query variants
  • hyde        — hypothetical document embeddings
  • filtered    — date/ACL/doc-type post-filtering
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from core.retrieval import Retriever, BM25Index
from core.rag_architectures import (
    SimpleRAG, MultiHopRAG, ConversationalRAG,
    AgenticRAG, CRAG, GraphRAG,
)

SYSTEM_PROMPT = (
    "You are TechCorp's AI assistant. Use the provided context to answer the question "
    "accurately and concisely. Answer based only on the provided context. "
    "If the context doesn't contain enough information, say so."
)


class ChatEngine:
    """
    Unified chat engine that wires LLM providers with RAG architectures
    and Section 4 retrieval strategies.
    """

    def __init__(
        self,
        vector_engine,
        architecture: str = "simple",
        retrieval_strategy: str = "top_k",
        n_results: int = 5,
        use_bm25: bool = True,
    ) -> None:
        self.vector_engine = vector_engine
        self.architecture = architecture
        self.retrieval_strategy = retrieval_strategy
        self.n_results = n_results
        self.provider = "fallback"
        self._setup_llm()

        # Build BM25 index if requested
        self._bm25: Optional[BM25Index] = None
        if use_bm25:
            self._bm25 = BM25Index()
            if vector_engine.is_initialized():
                try:
                    self._bm25.build(vector_engine)
                except Exception as e:
                    print(f"[ChatEngine] BM25 build failed: {e}")

        # Retriever wraps vector engine + BM25 + LLM (for multi-query/HyDE).
        # score_threshold=0.40 gives reasonable recall on a small demo corpus;
        # raise to 0.70 (SCORE_REJECT_THRESHOLD) for production with large indexes.
        self.retriever = Retriever(
            vector_engine=vector_engine,
            bm25_index=self._bm25,
            llm_fn=self._llm_fn,
            default_n=n_results,
            score_threshold=0.40,
        )

        # Architecture instances (built lazily or eagerly)
        self._conv_rag = ConversationalRAG(self.retriever, self._llm_fn, n_results=n_results)
        self._graph_rag: Optional[GraphRAG] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_response(
        self,
        question: str,
        architecture: Optional[str] = None,
        retrieval_strategy: Optional[str] = None,
        user_access_level: str = "public",
    ) -> Dict[str, Any]:
        """
        Main entry point.  Returns:
          {"answer", "sources", "confidence", "provider", "architecture", "retrieval_strategy"}
        """
        arch = architecture or self.architecture
        strategy = retrieval_strategy or self.retrieval_strategy

        try:
            raw = self._dispatch(question, arch, strategy, user_access_level)
        except Exception as e:
            raw = {
                "answer": f"An error occurred: {e}",
                "sources": [],
                "metadata": {"error": str(e)},
            }

        sources = raw.get("sources", [])
        # Normalise sources to match the existing API shape
        normalised_sources = []
        for s in sources:
            if isinstance(s, dict):
                meta = s.get("metadata", {})
                sim = s.get("similarity", s.get("rrf_score", 0.0))
                normalised_sources.append({
                    "category": meta.get("category", "General"),
                    "file": meta.get("file", meta.get("source", "document")),
                    "relevance": round(float(sim) * 100, 1) if sim else 0.0,
                })

        confidence = (
            round(sum(s["relevance"] for s in normalised_sources) / len(normalised_sources), 1)
            if normalised_sources else 0.0
        )

        return {
            "answer": raw.get("answer", ""),
            "sources": normalised_sources,
            "confidence": confidence,
            "provider": self.provider,
            "architecture": arch,
            "retrieval_strategy": strategy,
            "metadata": raw.get("metadata", {}),
        }

    def rebuild_bm25(self) -> None:
        """Rebuild the BM25 index after new documents are ingested."""
        if self._bm25 is not None:
            self._bm25.build(self.vector_engine)

    def reset_conversation(self) -> None:
        """Clear the conversational RAG memory."""
        self._conv_rag.reset()

    # ------------------------------------------------------------------
    # Architecture dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        question: str,
        arch: str,
        strategy: str,
        user_access_level: str,
    ) -> Dict[str, Any]:
        if arch == "multihop":
            return MultiHopRAG(
                self.retriever, self._llm_fn, max_hops=4
            ).run(question, user_access_level=user_access_level)

        if arch == "conversational":
            return self._conv_rag.run(question, user_access_level=user_access_level)

        if arch == "agentic":
            return AgenticRAG(
                self.retriever, self._llm_fn, max_steps=6
            ).run(question, user_access_level=user_access_level)

        if arch == "crag":
            return CRAG(
                self.retriever, self._llm_fn, n_results=self.n_results
            ).run(question, user_access_level=user_access_level)

        if arch == "graph":
            gr = self._get_graph_rag()
            return gr.run(question, user_access_level=user_access_level)

        # default: simple RAG
        return SimpleRAG(
            self.retriever, self._llm_fn, n_results=self.n_results
        ).run(question, strategy=strategy, user_access_level=user_access_level)

    def _get_graph_rag(self) -> GraphRAG:
        if self._graph_rag is None:
            self._graph_rag = GraphRAG(self.retriever, self._llm_fn)
            # Pre-build graph from existing documents
            try:
                result = self.vector_engine.collection.get(include=["documents"])
                docs = result.get("documents", [])
                if docs:
                    self._graph_rag.build_graph_from_documents(docs[:50])
            except Exception as e:
                print(f"[ChatEngine] Graph build failed: {e}")
        return self._graph_rag

    # ------------------------------------------------------------------
    # LLM callable wrapper
    # ------------------------------------------------------------------

    def _llm_fn(self, prompt: str) -> str:
        """Single callable passed to all architectures and retrieval strategies."""
        if self.provider == "gemini":
            return self._generate_with_gemini_raw(prompt)
        if self.provider == "anthropic":
            return self._generate_with_anthropic_raw(prompt)
        if self.provider == "openrouter":
            return self._generate_with_openrouter_raw(prompt)
        return self._generate_from_keywords(prompt)

    # ------------------------------------------------------------------
    # LLM setup (unchanged from original, extended with _raw helpers)
    # ------------------------------------------------------------------

    def _setup_llm(self) -> None:
        if self._try_gemini():
            return
        if self._try_anthropic():
            return
        if self._try_openrouter():
            return
        print("[ChatEngine] No LLM API key found — using keyword-extraction fallback")

    def _try_gemini(self) -> bool:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return False
        try:
            from google import genai
            self.gemini_client = genai.Client(api_key=api_key)
            self.provider = "gemini"
            print("[ChatEngine] Provider: Gemini (gemini-2.0-flash)")
            return True
        except ImportError:
            print("[ChatEngine] google-genai not installed — skipping Gemini")
            return False

    def _try_anthropic(self) -> bool:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return False
        try:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            self.provider = "anthropic"
            print("[ChatEngine] Provider: Anthropic (claude-haiku-4-5-20251001)")
            return True
        except ImportError:
            print("[ChatEngine] anthropic not installed — skipping Anthropic")
            return False

    def _try_openrouter(self) -> bool:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return False
        try:
            from openai import OpenAI
            self.openrouter_client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            self.openrouter_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            self.provider = "openrouter"
            print(f"[ChatEngine] Provider: OpenRouter ({self.openrouter_model})")
            return True
        except ImportError:
            print("[ChatEngine] openai package not installed — skipping OpenRouter")
            return False

    # ------------------------------------------------------------------
    # Raw LLM generators (accept prompt string, return answer string)
    # ------------------------------------------------------------------

    def _generate_with_gemini_raw(self, prompt: str) -> str:
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text

    def _generate_with_anthropic_raw(self, prompt: str) -> str:
        message = self.anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _generate_with_openrouter_raw(self, prompt: str) -> str:
        completion = self.openrouter_client.chat.completions.create(
            model=self.openrouter_model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def _generate_from_keywords(self, prompt: str) -> str:
        """Keyword-extraction fallback (no LLM required)."""
        # Extract the context portion from a built prompt
        if "Context:" in prompt:
            ctx_start = prompt.index("Context:") + len("Context:")
            q_start = prompt.index("Question:") if "Question:" in prompt else len(prompt)
            context = prompt[ctx_start:q_start].strip()
        else:
            context = prompt

        if not context.strip():
            return "I don't have enough information in the knowledge base to answer that question."

        lines = [l.strip() for l in context.split("\n") if l.strip()]
        return "Based on documents: " + " ".join(lines[:5])

    # ------------------------------------------------------------------
    # Backwards-compatible simple generator (used by app.py legacy routes)
    # ------------------------------------------------------------------

    def _generate_with_gemini(self, question: str, context: str) -> str:
        from core.foundations import build_rag_prompt
        return self._generate_with_gemini_raw(build_rag_prompt(question, context))

    def _generate_with_anthropic(self, question: str, context: str) -> str:
        from core.foundations import build_rag_prompt
        return self._generate_with_anthropic_raw(build_rag_prompt(question, context))

    def _generate_with_openrouter(self, question: str, context: str) -> str:
        from core.foundations import build_rag_prompt
        return self._generate_with_openrouter_raw(build_rag_prompt(question, context))

    def _generate_from_context(self, question: str, context: str) -> str:
        return self._generate_from_keywords(f"Context: {context}\nQuestion: {question}")
