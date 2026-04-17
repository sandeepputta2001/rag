"""
Chat Engine - generates answers using retrieved context (RAG)

Provider priority (first available wins):
  1. Gemini  — set GEMINI_API_KEY
  2. Anthropic (Claude) — set ANTHROPIC_API_KEY
  3. Keyword extraction fallback (no API key required)
"""

import os

SYSTEM_PROMPT = """You are TechCorp's AI assistant. Use the provided context to answer the question accurately and concisely.
Answer based only on the provided context. If the context doesn't contain enough information, say so."""


def _build_prompt(question: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

Context from TechCorp documents:
{context}

Question: {question}"""


class ChatEngine:
    def __init__(self, vector_engine):
        self.vector_engine = vector_engine
        self.provider = "fallback"  # "gemini" | "anthropic" | "fallback"
        self._setup_llm()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_llm(self):
        """Try Gemini first, then Anthropic, then fall back to keyword extraction."""
        if self._try_gemini():
            return
        if self._try_anthropic():
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_response(self, question: str) -> dict:
        results = self.vector_engine.search(question, n_results=3)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else []

        context = "\n\n".join(docs)

        sources = []
        for i, meta in enumerate(metas):
            confidence = round((1 - distances[i]) * 100, 1) if distances else 80.0
            sources.append({
                "category": meta.get("category", "General"),
                "file": meta.get("file", "document"),
                "relevance": confidence,
            })

        if self.provider == "gemini":
            answer = self._generate_with_gemini(question, context)
        elif self.provider == "anthropic":
            answer = self._generate_with_anthropic(question, context)
        else:
            answer = self._generate_from_context(question, context)

        overall_confidence = (
            round(sum(s["relevance"] for s in sources) / len(sources), 1) if sources else 0.0
        )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": overall_confidence,
            "provider": self.provider,
        }

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def _generate_with_gemini(self, question: str, context: str) -> str:
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=_build_prompt(question, context),
        )
        return response.text

    def _generate_with_anthropic(self, question: str, context: str) -> str:
        message = self.anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": _build_prompt(question, context)}],
        )
        return message.content[0].text

    def _generate_from_context(self, question: str, context: str) -> str:
        if not context.strip():
            return "I don't have enough information in the TechCorp knowledge base to answer that question."

        q_lower = question.lower()

        if any(w in q_lower for w in ["pet", "dog", "animal", "friday"]):
            lines = [l.strip() for l in context.split('\n') if any(w in l.lower() for w in ["pet", "dog", "friday", "vaccinated", "mascot"])]
        elif any(w in q_lower for w in ["remote", "work from home", "wfh", "days"]):
            lines = [l.strip() for l in context.split('\n') if any(w in l.lower() for w in ["remote", "days", "core hours", "meeting"])]
        elif any(w in q_lower for w in ["benefit", "health", "insurance", "401k", "pto", "vacation"]):
            lines = [l.strip() for l in context.split('\n') if any(w in l.lower() for w in ["health", "401k", "pto", "insurance", "dental", "vision", "learning"])]
        else:
            lines = [l.strip() for l in context.split('\n') if l.strip()][:5]

        relevant = [l for l in lines if l]
        if relevant:
            return "Based on TechCorp documents: " + " ".join(relevant[:4])

        return f"Based on TechCorp documents: {context[:300].strip()}..."
