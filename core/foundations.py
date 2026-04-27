"""
Section 1 – Foundations & Core Concepts

Implements:
  • Tokenisation (tiktoken BPE) with token counting & truncation
  • Context window budget planning (system + history + chunks + output)
  • Cosine similarity, dot product, euclidean distance with score labels
  • Prompt templates: simple RAG, chain-of-thought, few-shot, ReAct
  • Content-hash chunk ID generation for idempotent re-indexing
"""

import math
import hashlib
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Tokenisation — tiktoken BPE (cl100k_base = GPT-4 / text-embedding-3)
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False

_ENCODING = None  # loaded lazily on first use
TIKTOKEN_AVAILABLE = _TIKTOKEN_AVAILABLE


def _get_encoding(model: str = "cl100k_base"):
    """Return a tiktoken Encoding, loading it lazily and caching it."""
    global _ENCODING, TIKTOKEN_AVAILABLE
    if not _TIKTOKEN_AVAILABLE:
        return None
    if _ENCODING is not None and model == "cl100k_base":
        return _ENCODING
    try:
        enc = _tiktoken.get_encoding(model)
        if model == "cl100k_base":
            _ENCODING = enc
        return enc
    except Exception:
        # Network unavailable (vocab download needed), or other error
        TIKTOKEN_AVAILABLE = False
        return None


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count BPE tokens in text.
    Falls back to character/4 estimate when tiktoken is unavailable or offline.
    """
    enc = _get_encoding(model)
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def encode_tokens(text: str) -> List[int]:
    """Return the BPE token-id list for text (empty list if tiktoken absent/offline)."""
    enc = _get_encoding()
    return enc.encode(text) if enc else []


def truncate_to_tokens(text: str, max_tokens: int, model: str = "cl100k_base") -> str:
    """Return text truncated to at most max_tokens tokens."""
    enc = _get_encoding(model)
    if enc is not None:
        return enc.decode(enc.encode(text)[:max_tokens])
    return text[: max_tokens * 4]


# ---------------------------------------------------------------------------
# Context Window Budget Planner
# ---------------------------------------------------------------------------

CONTEXT_LIMITS: Dict[str, int] = {
    "gpt-4o":              128_000,
    "gpt-4o-mini":         128_000,
    "claude-3-5-sonnet":   200_000,
    "claude-haiku-4-5":    200_000,
    "gemini-2.0-flash":  1_048_576,
    "gemini-1.5-pro":    1_048_576,
    "default":              32_000,
}


def plan_context_budget(
    system_prompt: str,
    history: List[Dict[str, Any]],
    chunks: List[str],
    reserved_output: int = 1_024,
    model: str = "default",
) -> Dict[str, Any]:
    """
    Compute how many retrieved chunks fit inside the model's context window.

    Returns a breakdown dict with token counts per slot so callers can log it
    or display it in a debug UI.
    """
    limit = CONTEXT_LIMITS.get(model, CONTEXT_LIMITS["default"])
    system_tokens = count_tokens(system_prompt)
    history_tokens = sum(count_tokens(m.get("content", "")) for m in history)
    available = limit - system_tokens - history_tokens - reserved_output

    included_indices: List[int] = []
    running = 0
    chunk_token_counts = [count_tokens(c) for c in chunks]

    for i, ct in enumerate(chunk_token_counts):
        if running + ct > available:
            break
        included_indices.append(i)
        running += ct

    return {
        "model": model,
        "context_limit": limit,
        "system_tokens": system_tokens,
        "history_tokens": history_tokens,
        "reserved_output": reserved_output,
        "available_for_chunks": available,
        "chunks_requested": len(chunks),
        "chunks_included": len(included_indices),
        "chunk_tokens_included": running,
        "chunk_indices_included": included_indices,
        "fits_all": len(included_indices) == len(chunks),
    }


# ---------------------------------------------------------------------------
# Similarity Metrics
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity ∈ [-1, 1].  Returns 0 for zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# Thresholds from the RAG guide (cosine similarity scale)
SIMILARITY_THRESHOLDS: Dict[str, float] = {
    "very_high": 0.85,   # Strong semantic match — use directly
    "good":      0.70,   # Useful context — include
    "moderate":  0.50,   # Marginal — use only when nothing better exists
    "low":       0.30,   # Likely off-topic — discard
}

SCORE_REJECT_THRESHOLD = 0.70  # chunks below this are filtered out by default


def score_label(similarity: float) -> str:
    """Classify a cosine similarity score into a human-readable tier."""
    if similarity >= SIMILARITY_THRESHOLDS["very_high"]:
        return "very_high"
    if similarity >= SIMILARITY_THRESHOLDS["good"]:
        return "good"
    if similarity >= SIMILARITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "low"


def l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalise a vector (needed when using dot-product as cosine proxy)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

_BASE_SYSTEM = (
    "You are a knowledgeable assistant. "
    "Answer questions accurately and concisely using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Do not hallucinate or invent facts."
)


def build_rag_prompt(
    question: str,
    context: str,
    system: str = _BASE_SYSTEM,
) -> str:
    """Standard RAG prompt: system instruction + context + question."""
    return f"{system}\n\nContext:\n{context}\n\nQuestion: {question}"


def build_chain_of_thought_prompt(question: str, context: str) -> str:
    """
    Chain-of-thought variant.
    Instructs the model to reason step-by-step before giving a final answer.
    """
    cot_system = (
        _BASE_SYSTEM
        + "\n\nThink step by step before answering. "
        "Structure your response as:\n"
        "Reasoning: <your reasoning>\n"
        "Answer: <concise final answer>"
    )
    return build_rag_prompt(question, context, system=cot_system)


def build_few_shot_prompt(
    question: str,
    context: str,
    examples: List[Dict[str, str]],
) -> str:
    """
    Few-shot prompt.
    Prepends Q&A examples so the model learns the expected response style.
    examples = [{"question": "...", "answer": "..."}, ...]
    """
    example_block = "\n\n".join(
        f"Q: {e['question']}\nA: {e['answer']}" for e in examples
    )
    few_shot_system = (
        _BASE_SYSTEM
        + f"\n\nHere are examples of ideal answers:\n{example_block}"
    )
    return build_rag_prompt(question, context, system=few_shot_system)


def build_react_prompt(
    question: str,
    available_tools: List[str],
    previous_steps: str = "",
) -> str:
    """
    ReAct (Reason + Act) prompt for agentic RAG.
    The model fills in a Thought → Action → Observation loop.
    """
    tools_str = "\n".join(f"  - {t}" for t in available_tools)
    return (
        "You are an AI agent that solves problems via a Thought → Action → Observation loop.\n\n"
        f"Available tools:\n{tools_str}\n\n"
        f"Question: {question}\n\n"
        f"{previous_steps}"
        "Thought:"
    )


def build_hyde_prompt(question: str) -> str:
    """
    HyDE — prompt that generates a hypothetical answer document.
    The resulting text is embedded and used as the search query.
    """
    return (
        "Write a detailed passage that would directly answer the following question. "
        "Write as if you are an expert authoring a knowledge-base article.\n\n"
        f"Question: {question}\n\n"
        "Passage:"
    )


def build_query_expansion_prompt(question: str, n: int = 3) -> str:
    """
    Multi-query expansion prompt.
    Asks the model to generate n alternative phrasings of the question.
    """
    return (
        f"Generate {n} different ways to ask the following question. "
        "Each variant should capture the same intent but use different vocabulary or framing. "
        "Output only the variants, one per line, no numbering.\n\n"
        f"Original question: {question}"
    )


def build_query_contextualise_prompt(
    question: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """
    Conversational RAG — reformulate a follow-up question into a standalone question
    that makes sense without the conversation history.
    """
    history_str = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
    )
    return (
        "Given the conversation history below and a follow-up question, "
        "rewrite the follow-up question as a fully self-contained standalone question.\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )


def build_relevance_grading_prompt(question: str, chunk: str) -> str:
    """
    CRAG relevance grading.
    Returns RELEVANT or NOT RELEVANT so callers can decide whether to fall back to web search.
    """
    return (
        "Given a question and a context passage, determine whether the passage is "
        "relevant enough to help answer the question.\n\n"
        f"Question: {question}\n\n"
        f"Passage:\n{chunk}\n\n"
        "Respond with exactly one word: RELEVANT or NOT_RELEVANT."
    )


# ---------------------------------------------------------------------------
# Chunk ID — deterministic content-hash for idempotent re-indexing
# ---------------------------------------------------------------------------

def make_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """
    Build a stable, unique chunk ID from the source path, index, and content hash.
    Re-ingesting the same document always yields the same IDs — prevents duplicates.
    """
    h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
    safe = source.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{safe}__c{chunk_index:04d}__{h}"


# ---------------------------------------------------------------------------
# RAG vs Fine-Tuning decision helper (Section 1.1)
# ---------------------------------------------------------------------------

def should_use_rag(
    knowledge_changes_frequently: bool,
    needs_source_attribution: bool,
    data_is_private: bool,
    task_requires_style_change: bool,
) -> Dict[str, Any]:
    """
    Simple decision tree: RAG vs Fine-Tuning vs Both.
    Returns a recommendation with reasoning.
    """
    if knowledge_changes_frequently or needs_source_attribution or data_is_private:
        if task_requires_style_change:
            return {
                "recommendation": "Both",
                "reasoning": (
                    "Dynamic/private knowledge suggests RAG; "
                    "style or format changes suggest fine-tuning. "
                    "Use both for best results."
                ),
            }
        return {
            "recommendation": "RAG",
            "reasoning": (
                "Frequently updated, private, or attribution-required knowledge "
                "is best served by RAG — no retraining needed and sources are citable."
            ),
        }
    if task_requires_style_change:
        return {
            "recommendation": "Fine-Tuning",
            "reasoning": (
                "Stable knowledge with a clear style or format requirement "
                "is a good candidate for fine-tuning."
            ),
        }
    return {
        "recommendation": "RAG",
        "reasoning": "RAG is the safer default: no training cost and knowledge is updatable at any time.",
    }
