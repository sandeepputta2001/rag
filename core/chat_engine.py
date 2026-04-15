"""
Chat Engine - generates answers using retrieved context (RAG)
"""

import os


class ChatEngine:
    def __init__(self, vector_engine):
        self.vector_engine = vector_engine
        self._setup_llm()

    def _setup_llm(self):
        self.use_claude = False
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                self.use_claude = True
            except ImportError:
                pass

    def get_response(self, question):
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
                "relevance": confidence
            })

        if self.use_claude:
            answer = self._generate_with_claude(question, context)
        else:
            answer = self._generate_from_context(question, context)

        overall_confidence = round(sum(s["relevance"] for s in sources) / len(sources), 1) if sources else 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": overall_confidence
        }

    def _generate_with_claude(self, question, context):
        prompt = f"""You are TechCorp's AI assistant. Use the provided context to answer the question accurately and concisely.

Context from TechCorp documents:
{context}

Question: {question}

Answer based only on the provided context. If the context doesn't contain enough information, say so."""

        message = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _generate_from_context(self, question, context):
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

        first_300 = context[:300].strip()
        return f"Based on TechCorp documents: {first_300}..."
