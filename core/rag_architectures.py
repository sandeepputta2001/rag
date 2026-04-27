"""
Section 5 – RAG Architectures

Implements (all as composable classes around a Retriever + LLM callable):

  • SimpleRAG        — standard ingest → retrieve → prompt → generate pipeline
  • MultiHopRAG      — iterative Plan-Retrieve-Reason loop until DONE
  • ConversationalRAG— query contextualisation (follow-up → standalone) + memory
  • AgenticRAG       — ReAct (Reason + Act) loop with pluggable tool registry
  • CRAG             — self-correcting: relevance-grade chunks, web-search fallback
  • GraphRAG          — extract (subject, relation, object) triples, build
                        NetworkX DiGraph, answer via ego-graph traversal

All architectures share the same interface:
    result = Architecture(retriever, llm_fn).run(question, **kwargs)
    → {"answer": str, "sources": list[dict], "metadata": dict}
"""

from __future__ import annotations

import re
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.foundations import (
    build_rag_prompt,
    build_chain_of_thought_prompt,
    build_react_prompt,
    build_query_contextualise_prompt,
    build_relevance_grading_prompt,
    build_hyde_prompt,
    plan_context_budget,
)

LLMFn = Callable[[str], str]


def _src(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a retrieval hit into a compact source dict with a unified score."""
    return {
        "text": hit["text"][:200],
        "metadata": hit.get("metadata", {}),
        "similarity": hit.get("similarity", hit.get("rrf_score", 0.0)),
    }


# ---------------------------------------------------------------------------
# Architecture 1: Simple RAG
# ---------------------------------------------------------------------------

class SimpleRAG:
    """
    Standard single-hop RAG: retrieve → prompt → generate.
    Supports chain-of-thought mode.
    """

    def __init__(
        self,
        retriever,
        llm_fn: LLMFn,
        n_results: int = 5,
        use_cot: bool = False,
    ) -> None:
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.n_results = n_results
        self.use_cot = use_cot

    def run(
        self,
        question: str,
        strategy: str = "top_k",
        user_access_level: str = "public",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        hits = self.retriever.retrieve(
            question,
            strategy=strategy,
            n=self.n_results,
            user_access_level=user_access_level,
            **kwargs,
        )
        context = "\n\n".join(h["text"] for h in hits)

        prompt = (
            build_chain_of_thought_prompt(question, context)
            if self.use_cot
            else build_rag_prompt(question, context)
        )
        answer = self.llm_fn(prompt)

        return {
            "answer": answer,
            "sources": [_src(h) for h in hits],
            "metadata": {"architecture": "simple_rag", "strategy": strategy, "n_chunks": len(hits)},
        }


# ---------------------------------------------------------------------------
# Architecture 2: Multi-Hop RAG
# ---------------------------------------------------------------------------

_MULTIHOP_SYSTEM = """You are a multi-step reasoning assistant.
Your goal is to answer a complex question by breaking it into sub-questions,
retrieving evidence for each sub-question, and synthesising a final answer.

At each step respond in exactly this JSON format (no markdown):
{"thought": "...", "action": "search", "query": "..."}
OR when you have enough information:
{"thought": "...", "action": "done", "answer": "..."}
"""


class MultiHopRAG:
    """
    Iterative Plan-Retrieve-Reason loop.

    Each iteration:
      1. LLM produces a {thought, action, query} JSON.
      2. If action == "search": retrieve chunks for the sub-query.
      3. Accumulate evidence.
      4. If action == "done" or max_hops reached: synthesise final answer.
    """

    def __init__(
        self,
        retriever,
        llm_fn: LLMFn,
        max_hops: int = 4,
        n_per_hop: int = 3,
    ) -> None:
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.max_hops = max_hops
        self.n_per_hop = n_per_hop

    def run(self, question: str, user_access_level: str = "public") -> Dict[str, Any]:
        history_text = ""
        all_evidence: List[Dict[str, Any]] = []
        hops_taken = 0

        for hop in range(self.max_hops):
            prompt = (
                f"{_MULTIHOP_SYSTEM}\n\n"
                f"Original question: {question}\n\n"
                f"Evidence so far:\n{_evidence_summary(all_evidence)}\n\n"
                f"{history_text}"
                "Next step (JSON only):"
            )
            raw = self.llm_fn(prompt).strip()
            step = _parse_json_safe(raw)
            hops_taken += 1

            action = step.get("action", "")
            history_text += f"Thought: {step.get('thought', '')}\n"

            if action == "done" or (not action and "answer" in step):
                answer = step.get("answer", raw)
                return {
                    "answer": answer,
                    "sources": [_src(e) for e in all_evidence],
                    "metadata": {"architecture": "multi_hop_rag", "hops": hops_taken},
                }

            sub_query = step.get("query", question)
            hits = self.retriever.retrieve(
                sub_query, strategy="top_k", n=self.n_per_hop,
                user_access_level=user_access_level,
            )
            all_evidence.extend(hits)
            history_text += f"Action: search('{sub_query}')\n"
            history_text += f"Observation: {len(hits)} chunks retrieved\n\n"

        # Max hops reached — synthesise with accumulated evidence
        context = "\n\n".join(e["text"] for e in all_evidence)
        final = self.llm_fn(build_rag_prompt(question, context))
        return {
            "answer": final,
            "sources": [_src(e) for e in all_evidence],
            "metadata": {"architecture": "multi_hop_rag", "hops": hops_taken, "max_hops_reached": True},
        }


# ---------------------------------------------------------------------------
# Architecture 3: Conversational RAG
# ---------------------------------------------------------------------------

class ConversationalRAG:
    """
    Conversational RAG with:
    • Chat history management (sliding window to avoid context overflow)
    • Query contextualisation: rewrites follow-up questions into standalone form
    • Memory: stores (question, answer) pairs
    """

    def __init__(
        self,
        retriever,
        llm_fn: LLMFn,
        n_results: int = 5,
        history_window: int = 6,
    ) -> None:
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.n_results = n_results
        self.history_window = history_window
        self.chat_history: List[Dict[str, str]] = []

    def run(self, question: str, user_access_level: str = "public") -> Dict[str, Any]:
        # Step 1: contextualise follow-up into standalone question
        standalone = question
        if self.chat_history:
            ctx_prompt = build_query_contextualise_prompt(question, self.chat_history)
            try:
                standalone = self.llm_fn(ctx_prompt).strip()
                print(f"[ConversationalRAG] Standalone: {standalone}")
            except Exception:
                standalone = question

        # Step 2: retrieve
        hits = self.retriever.retrieve(
            standalone, strategy="top_k", n=self.n_results,
            user_access_level=user_access_level,
        )
        context = "\n\n".join(h["text"] for h in hits)

        # Step 3: generate (inject history for conversational awareness)
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in self.chat_history[-self.history_window:]
        )
        augmented_context = (
            f"Conversation so far:\n{history_str}\n\n"
            f"Retrieved knowledge:\n{context}"
            if history_str else context
        )

        answer = self.llm_fn(build_rag_prompt(question, augmented_context))

        # Update memory
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})
        # Trim to window
        if len(self.chat_history) > self.history_window * 2:
            self.chat_history = self.chat_history[-self.history_window * 2:]

        return {
            "answer": answer,
            "standalone_question": standalone,
            "sources": [_src(h) for h in hits],
            "metadata": {
                "architecture": "conversational_rag",
                "history_turns": len(self.chat_history) // 2,
            },
        }

    def reset(self) -> None:
        """Clear conversation memory."""
        self.chat_history = []


# ---------------------------------------------------------------------------
# Architecture 4: Agentic RAG (ReAct loop)
# ---------------------------------------------------------------------------

class AgenticRAG:
    """
    ReAct (Reason + Act) agentic loop.

    The LLM has access to a pluggable tool registry.  Built-in tools:
      • search(query)    — vector store retrieval
      • summarise(text)  — extract key points from a long document
      • finish(answer)   — terminate the loop with a final answer

    Custom tools can be injected via the tools dict:
        {"tool_name": callable_fn}
    """

    def __init__(
        self,
        retriever,
        llm_fn: LLMFn,
        tools: Optional[Dict[str, Callable]] = None,
        max_steps: int = 8,
        n_per_search: int = 4,
    ) -> None:
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.max_steps = max_steps
        self.n_per_search = n_per_search

        # Built-in tool implementations
        self._tools: Dict[str, Callable] = {
            "search": self._tool_search,
            "summarise": self._tool_summarise,
            "finish": lambda answer: answer,
        }
        if tools:
            self._tools.update(tools)

    def run(self, question: str, user_access_level: str = "public") -> Dict[str, Any]:
        tool_names = list(self._tools.keys())
        steps_log: List[Dict[str, Any]] = []
        trajectory = ""
        self._access_level = user_access_level

        for step in range(self.max_steps):
            prompt = build_react_prompt(question, tool_names, previous_steps=trajectory)
            raw = self.llm_fn(prompt).strip()

            # Parse Thought / Action / Action Input from the raw output
            thought, action, action_input = _parse_react_step(raw)
            steps_log.append({"thought": thought, "action": action, "input": action_input})

            if action.lower() in ("finish", "done"):
                return {
                    "answer": action_input or thought,
                    "sources": [],
                    "metadata": {"architecture": "agentic_rag", "steps": step + 1, "log": steps_log},
                }

            # Execute the tool
            tool_fn = self._tools.get(action.lower())
            if tool_fn:
                try:
                    observation = tool_fn(action_input)
                except Exception as e:
                    observation = f"Error: {e}"
            else:
                observation = f"Unknown tool '{action}'. Available: {tool_names}"

            trajectory += (
                f"Thought: {thought}\n"
                f"Action: {action}[{action_input}]\n"
                f"Observation: {str(observation)[:500]}\n\n"
            )

        # Max steps reached
        final_prompt = f"Based on the following research, answer: {question}\n\n{trajectory}"
        answer = self.llm_fn(final_prompt)
        return {
            "answer": answer,
            "sources": [],
            "metadata": {"architecture": "agentic_rag", "steps": self.max_steps,
                         "max_steps_reached": True, "log": steps_log},
        }

    def _tool_search(self, query: str) -> str:
        hits = self.retriever.retrieve(
            query, strategy="top_k", n=self.n_per_search,
            user_access_level=getattr(self, "_access_level", "public"),
        )
        if not hits:
            return "No relevant documents found."
        parts = []
        for h in hits:
            src = h.get("metadata", {}).get("source", "unknown")
            parts.append(f"[{src}]\n{h['text'][:400]}")
        return "\n\n---\n\n".join(parts)

    def _tool_summarise(self, text: str) -> str:
        return self.llm_fn(
            f"Summarise the following text in 3–5 bullet points:\n\n{text[:2000]}"
        )


# ---------------------------------------------------------------------------
# Architecture 5: CRAG — Self-Correcting RAG
# ---------------------------------------------------------------------------

class CRAG:
    """
    Corrective RAG (Shi et al., 2023).

    For each retrieved chunk:
    1. Grade relevance with the LLM.
    2. If any chunk is RELEVANT → answer from relevant chunks.
    3. If ALL chunks are NOT_RELEVANT → trigger web-search fallback (or
       expand retrieval if no web-search fn is available).
    4. Blend: if some relevant + some not → use relevant + fall back for rest.
    """

    def __init__(
        self,
        retriever,
        llm_fn: LLMFn,
        web_search_fn: Optional[Callable[[str], str]] = None,
        n_results: int = 5,
        relevance_threshold: float = 0.0,
    ) -> None:
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.web_search_fn = web_search_fn
        self.n_results = n_results
        self.relevance_threshold = relevance_threshold

    def run(self, question: str, user_access_level: str = "public") -> Dict[str, Any]:
        hits = self.retriever.retrieve(
            question, strategy="top_k", n=self.n_results,
            user_access_level=user_access_level,
        )

        # Grade each chunk
        relevant, not_relevant = [], []
        for hit in hits:
            grade = self._grade(question, hit["text"])
            hit["relevance_grade"] = grade
            (relevant if grade == "RELEVANT" else not_relevant).append(hit)

        print(f"[CRAG] Relevant: {len(relevant)} / Not relevant: {len(not_relevant)}")

        if relevant:
            context_chunks = relevant
        elif self.web_search_fn:
            # All retrieved chunks are off-topic → web search
            web_text = self.web_search_fn(question)
            context_chunks = [{"text": web_text, "metadata": {"source": "web_search"},
                                "similarity": 0.0}]
        else:
            # No web search — fall back to keyword-expanded retrieval
            expanded_hits = self.retriever.retrieve(
                question, strategy="multi_query", n=self.n_results,
                user_access_level=user_access_level,
            )
            context_chunks = expanded_hits or hits

        context = "\n\n".join(c["text"] for c in context_chunks)
        answer = self.llm_fn(build_rag_prompt(question, context))

        return {
            "answer": answer,
            "sources": [{**_src(c), "relevance_grade": c.get("relevance_grade", "?")}
                        for c in context_chunks],
            "metadata": {
                "architecture": "crag",
                "relevant": len(relevant),
                "not_relevant": len(not_relevant),
                "web_fallback_used": not relevant and bool(self.web_search_fn),
            },
        }

    def _grade(self, question: str, chunk: str) -> str:
        prompt = build_relevance_grading_prompt(question, chunk)
        try:
            verdict = self.llm_fn(prompt).strip().upper()
            # Normalise: accept RELEVANT / NOT_RELEVANT / NOT RELEVANT
            if "NOT" in verdict:
                return "NOT_RELEVANT"
            return "RELEVANT"
        except Exception:
            return "RELEVANT"  # fail-safe: treat as relevant


# ---------------------------------------------------------------------------
# Architecture 6: Graph RAG (NetworkX)
# ---------------------------------------------------------------------------

class GraphRAG:
    """
    Graph RAG: extracts (subject, relation, object) triples from text,
    stores them in a NetworkX DiGraph, and answers via ego-graph traversal.

    Triple extraction uses the LLM (prompt-based) since we don't require
    spaCy / custom NER here.
    """

    def __init__(
        self,
        retriever,
        llm_fn: LLMFn,
        n_results: int = 5,
        graph_radius: int = 2,
    ) -> None:
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.n_results = n_results
        self.graph_radius = graph_radius

        try:
            import networkx as nx  # type: ignore
            self._graph: Any = nx.DiGraph()
            self._nx = nx
            self._nx_available = True
        except ImportError:
            self._nx_available = False
            print("[GraphRAG] networkx not installed — graph operations disabled")

    # ------------------------------------------------------------------

    def build_graph_from_documents(self, texts: List[str]) -> None:
        """Extract triples from texts and populate the knowledge graph."""
        if not self._nx_available:
            return
        total_triples = 0
        for text in texts:
            triples = self._extract_triples(text)
            for subj, rel, obj in triples:
                self._graph.add_edge(subj, obj, relation=rel)
            total_triples += len(triples)
        print(f"[GraphRAG] Graph: {self._graph.number_of_nodes()} nodes, "
              f"{self._graph.number_of_edges()} edges from {len(texts)} texts")

    def run(self, question: str, user_access_level: str = "public") -> Dict[str, Any]:
        # Step 1: retrieve seed chunks
        hits = self.retriever.retrieve(
            question, strategy="top_k", n=self.n_results,
            user_access_level=user_access_level,
        )

        if not self._nx_available or self._graph.number_of_nodes() == 0:
            # No graph available — fall back to simple RAG
            context = "\n\n".join(h["text"] for h in hits)
            answer = self.llm_fn(build_rag_prompt(question, context))
            return {
                "answer": answer,
                "sources": [_src(h) for h in hits],
                "metadata": {"architecture": "graph_rag", "graph_used": False},
            }

        # Step 2: identify key entities in the question
        entities = self._extract_entities(question)

        # Step 3: expand via ego-graph traversal
        graph_context_parts: List[str] = []
        for entity in entities:
            # Find the best-matching node
            node = self._find_node(entity)
            if node is None:
                continue
            subgraph = self._nx.ego_graph(self._graph, node, radius=self.graph_radius)
            for u, v, data in subgraph.edges(data=True):
                graph_context_parts.append(
                    f"{u} --[{data.get('relation', 'related_to')}]--> {v}"
                )

        graph_context = "\n".join(graph_context_parts[:50])  # cap at 50 triples
        retrieval_context = "\n\n".join(h["text"] for h in hits)

        combined = (
            f"Knowledge graph facts:\n{graph_context}\n\n"
            f"Retrieved passages:\n{retrieval_context}"
        )

        answer = self.llm_fn(build_rag_prompt(question, combined))

        return {
            "answer": answer,
            "sources": [_src(h) for h in hits],
            "graph_triples": graph_context_parts[:20],
            "metadata": {
                "architecture": "graph_rag",
                "graph_used": True,
                "entities_found": entities,
                "graph_nodes": self._graph.number_of_nodes(),
                "graph_edges": self._graph.number_of_edges(),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Ask the LLM to extract (subject, relation, object) triples from text.
        Returns a list of 3-tuples.
        """
        prompt = (
            "Extract factual (subject, relation, object) triples from the text below. "
            "Output ONE triple per line in the format: subject | relation | object\n"
            "Use concise, normalised entity names. Extract at most 10 triples.\n\n"
            f"Text:\n{text[:1500]}\n\nTriples:"
        )
        try:
            raw = self.llm_fn(prompt)
            triples = []
            for line in raw.strip().split("\n"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3 and all(parts):
                    triples.append(tuple(parts))  # type: ignore[arg-type]
            return triples
        except Exception:
            return []

    def _extract_entities(self, question: str) -> List[str]:
        """Extract key entities from the question using the LLM."""
        prompt = (
            "List the key entities (people, organisations, concepts, products) "
            "mentioned in this question. Output one entity per line.\n\n"
            f"Question: {question}\n\nEntities:"
        )
        try:
            raw = self.llm_fn(prompt)
            return [line.strip() for line in raw.strip().split("\n") if line.strip()][:5]
        except Exception:
            return []

    def _find_node(self, entity: str) -> Optional[str]:
        """Find the best-matching node in the graph (case-insensitive substring)."""
        entity_lower = entity.lower()
        for node in self._graph.nodes():
            if entity_lower in str(node).lower() or str(node).lower() in entity_lower:
                return node
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence_summary(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "(none yet)"
    return "\n\n".join(f"[{i+1}] {e['text'][:300]}" for i, e in enumerate(evidence[:5]))


def _parse_json_safe(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a string, tolerating surrounding text."""
    text = text.strip()
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Fallback: treat as done with the raw text as answer
    return {"action": "done", "answer": text}


def _parse_react_step(raw: str) -> Tuple[str, str, str]:
    """
    Parse a ReAct step from LLM output.
    Looks for lines starting with "Thought:", "Action:", "Action Input:".
    Returns (thought, action, action_input).
    """
    thought = action = action_input = ""
    for line in raw.split("\n"):
        l = line.strip()
        if l.startswith("Thought:"):
            thought = l[len("Thought:"):].strip()
        elif l.startswith("Action:"):
            # Action: tool_name[input] OR Action: tool_name
            a = l[len("Action:"):].strip()
            m = re.match(r"(\w+)\[?(.*?)\]?$", a)
            if m:
                action = m.group(1)
                action_input = m.group(2).strip()
            else:
                action = a
        elif l.startswith("Action Input:"):
            action_input = l[len("Action Input:"):].strip()

    # If model wrote "finish" or "done" in thought without explicit action
    if not action:
        if any(w in raw.lower() for w in ["final answer", "answer:", "finish"]):
            action = "finish"
            # Try to extract the answer text
            for kw in ["Final Answer:", "Answer:", "Finish["]:
                if kw.lower() in raw.lower():
                    idx = raw.lower().index(kw.lower()) + len(kw)
                    action_input = raw[idx:].strip().strip("]").strip()
                    break
            if not action_input:
                action_input = thought

    return thought, action, action_input
