"""
Section 5 – Baseline Evaluation

Implements:
  • GoldenDataset  — manage 50–100 Q&A pairs with expected answers & source refs
  • RAGEvaluator   — automated scoring (retrieval relevance, faithfulness proxy,
                     exact-match, BLEU-1, answer length) + human eval checklist
  • EvalReport     — structured report with per-question and aggregate metrics
  • Logging to JSONL for persistence and CI integration
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GoldenExample:
    """A single ground-truth Q&A pair in the evaluation dataset."""
    question: str
    expected_answer: str
    expected_sources: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"   # easy | medium | hard
    notes: str = ""


@dataclass
class QuestionResult:
    """Evaluation result for a single question."""
    question: str
    expected_answer: str
    actual_answer: str
    retrieved_chunks: List[str]
    retrieved_sources: List[str]
    latency_ms: float

    # Automated metrics (filled in by RAGEvaluator)
    retrieval_relevance: float = 0.0   # fraction of retrieved chunks marked relevant
    faithfulness_score: float = 0.0    # fraction of answer sentences grounded in context
    exact_match: bool = False
    bleu1: float = 0.0
    answer_length: int = 0

    # Human eval (filled in via human_eval())
    human_retrieval_ok: Optional[bool] = None
    human_faithfulness_ok: Optional[bool] = None
    human_completeness_ok: Optional[bool] = None
    human_no_hallucination: Optional[bool] = None
    human_notes: str = ""


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    run_id: str
    timestamp: str
    architecture: str
    retrieval_strategy: str
    n_questions: int

    avg_retrieval_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    exact_match_rate: float = 0.0
    avg_bleu1: float = 0.0
    avg_latency_ms: float = 0.0

    human_retrieval_ok_rate: Optional[float] = None
    human_faithfulness_ok_rate: Optional[float] = None
    human_no_hallucination_rate: Optional[float] = None

    per_question: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Golden Dataset manager
# ---------------------------------------------------------------------------

SAMPLE_GOLDEN_DATASET: List[GoldenExample] = [
    GoldenExample(
        question="What is RAG and why is it used?",
        expected_answer="RAG (Retrieval-Augmented Generation) combines information retrieval with "
                        "language generation to ground LLM responses in factual documents.",
        category="concepts",
        difficulty="easy",
    ),
    GoldenExample(
        question="What is the difference between fixed chunking and recursive chunking?",
        expected_answer="Fixed chunking splits text into equal-size windows with overlap. "
                        "Recursive chunking tries structural separators (paragraph, sentence, word) "
                        "before falling back to character splits.",
        category="data_ingestion",
        difficulty="medium",
    ),
    GoldenExample(
        question="How does Reciprocal Rank Fusion combine BM25 and vector search?",
        expected_answer="RRF assigns each document a score of 1/(k+rank) in each ranked list "
                        "(k=60 by default), then sums scores across lists to produce a merged ranking.",
        category="retrieval",
        difficulty="hard",
    ),
    GoldenExample(
        question="What is HyDE and when should it be used?",
        expected_answer="HyDE (Hypothetical Document Embeddings) generates a hypothetical answer "
                        "to the query and embeds that instead of the raw query. It improves retrieval "
                        "for vague or abstract questions.",
        category="retrieval",
        difficulty="medium",
    ),
    GoldenExample(
        question="What cosine similarity threshold should trigger chunk rejection?",
        expected_answer="Chunks with cosine similarity below 0.70 are generally considered "
                        "off-topic and should be rejected.",
        category="retrieval",
        difficulty="easy",
    ),
]


class GoldenDataset:
    """Load, save, and manage the golden Q&A evaluation dataset."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self.examples: List[GoldenExample] = []

    def load(self, path: Optional[str] = None) -> "GoldenDataset":
        target = Path(path) if path else self.path
        if target and target.exists():
            with target.open() as f:
                raw = json.load(f)
            self.examples = [GoldenExample(**r) for r in raw]
            print(f"[GoldenDataset] Loaded {len(self.examples)} examples from {target}")
        else:
            self.examples = list(SAMPLE_GOLDEN_DATASET)
            print(f"[GoldenDataset] Using built-in sample dataset ({len(self.examples)} examples)")
        return self

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("No path provided for saving the golden dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as f:
            json.dump([asdict(e) for e in self.examples], f, indent=2)
        print(f"[GoldenDataset] Saved {len(self.examples)} examples to {target}")

    def add(self, example: GoldenExample) -> None:
        self.examples.append(example)

    def filter_by(self, category: Optional[str] = None, difficulty: Optional[str] = None) -> List[GoldenExample]:
        out = self.examples
        if category:
            out = [e for e in out if e.category == category]
        if difficulty:
            out = [e for e in out if e.difficulty == difficulty]
        return out

    def __len__(self) -> int:
        return len(self.examples)


# ---------------------------------------------------------------------------
# Automated scoring helpers
# ---------------------------------------------------------------------------

def _bleu1(reference: str, hypothesis: str) -> float:
    """Unigram BLEU (BLEU-1) — fraction of hypothesis tokens found in reference."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    return round(matches / len(hyp_tokens), 4)


def _sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _faithfulness_score(answer: str, context: str) -> float:
    """
    Heuristic faithfulness proxy.
    Counts what fraction of answer sentences have ≥1 word overlap with context.
    (A proper implementation would use an LLM judge or NLI model.)
    """
    context_words = set(context.lower().split())
    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0
    grounded = sum(
        1 for s in sentences
        if any(w in context_words for w in s.lower().split() if len(w) > 4)
    )
    return round(grounded / len(sentences), 4)


def _retrieval_relevance(question: str, chunks: List[str]) -> float:
    """
    Keyword-overlap heuristic: fraction of retrieved chunks that share ≥2 keywords
    with the question.  Replace with an LLM judge for production.
    """
    q_words = {w for w in re.sub(r"[^\w\s]", " ", question.lower()).split() if len(w) > 3}
    if not chunks or not q_words:
        return 0.0
    relevant = sum(
        1 for c in chunks
        if len(q_words & {w for w in c.lower().split() if len(w) > 3}) >= 2
    )
    return round(relevant / len(chunks), 4)


# ---------------------------------------------------------------------------
# RAG Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Runs the full evaluation loop over a GoldenDataset and generates an EvalReport.

    rag_fn: callable(question: str) → {"answer": str, "sources": list[dict], ...}
    llm_judge_fn: optional LLM callable for relevance grading (enhances faithfulness score)
    """

    def __init__(
        self,
        rag_fn: Callable[[str], Dict[str, Any]],
        llm_judge_fn: Optional[Callable[[str], str]] = None,
        log_dir: str = "./eval_logs",
    ) -> None:
        self.rag_fn = rag_fn
        self.llm_judge_fn = llm_judge_fn
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        dataset: GoldenDataset,
        architecture: str = "simple_rag",
        retrieval_strategy: str = "top_k",
        subset: Optional[int] = None,
    ) -> EvalReport:
        """
        Run all questions in the dataset through the RAG function and score results.
        subset: if set, evaluate only the first N examples (useful for fast CI runs).
        """
        examples = dataset.examples[:subset] if subset else dataset.examples
        run_id = f"eval_{int(time.time())}"
        results: List[QuestionResult] = []

        for i, ex in enumerate(examples):
            print(f"  [{i+1}/{len(examples)}] {ex.question[:60]}...")
            result = self._evaluate_one(ex)
            results.append(result)

        report = self._aggregate(results, run_id, architecture, retrieval_strategy)
        self._log(report, run_id)
        return report

    def _evaluate_one(self, ex: GoldenExample) -> QuestionResult:
        t0 = time.perf_counter()
        try:
            response = self.rag_fn(ex.question)
        except Exception as e:
            response = {"answer": f"ERROR: {e}", "sources": []}
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        actual = response.get("answer", "")
        sources_raw = response.get("sources", [])
        chunks = [s.get("text", "") for s in sources_raw]
        source_ids = [s.get("metadata", {}).get("source", "") for s in sources_raw]

        qr = QuestionResult(
            question=ex.question,
            expected_answer=ex.expected_answer,
            actual_answer=actual,
            retrieved_chunks=chunks,
            retrieved_sources=source_ids,
            latency_ms=latency_ms,
        )

        # Automated metrics
        ctx = "\n".join(chunks)
        qr.retrieval_relevance = _retrieval_relevance(ex.question, chunks)
        qr.faithfulness_score = _faithfulness_score(actual, ctx)
        qr.exact_match = ex.expected_answer.lower().strip() == actual.lower().strip()
        qr.bleu1 = _bleu1(ex.expected_answer, actual)
        qr.answer_length = len(actual.split())

        return qr

    def _aggregate(
        self,
        results: List[QuestionResult],
        run_id: str,
        architecture: str,
        retrieval_strategy: str,
    ) -> EvalReport:
        n = len(results)
        if n == 0:
            return EvalReport(
                run_id=run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                architecture=architecture,
                retrieval_strategy=retrieval_strategy,
                n_questions=0,
            )

        report = EvalReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            architecture=architecture,
            retrieval_strategy=retrieval_strategy,
            n_questions=n,
            avg_retrieval_relevance=round(sum(r.retrieval_relevance for r in results) / n, 4),
            avg_faithfulness=round(sum(r.faithfulness_score for r in results) / n, 4),
            exact_match_rate=round(sum(r.exact_match for r in results) / n, 4),
            avg_bleu1=round(sum(r.bleu1 for r in results) / n, 4),
            avg_latency_ms=round(sum(r.latency_ms for r in results) / n, 2),
            per_question=[asdict(r) for r in results],
        )
        return report

    def _log(self, report: EvalReport, run_id: str) -> None:
        log_path = self.log_dir / f"{run_id}.json"
        with log_path.open("w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"[RAGEvaluator] Report saved → {log_path}")

    # ------------------------------------------------------------------
    # Human evaluation checklist
    # ------------------------------------------------------------------

    def human_eval(
        self,
        results: List[QuestionResult],
        interactive: bool = False,
    ) -> List[QuestionResult]:
        """
        Human evaluation checklist (Section 5.6).

        Dimensions:
          1. Retrieval relevance — are the retrieved chunks relevant?
          2. Faithfulness       — does the answer stay within the retrieved context?
          3. Completeness       — does the answer address all parts of the question?
          4. No hallucination   — does the answer introduce facts not in context?

        If interactive=True, prints each result and prompts for Y/N input.
        Otherwise, fills in None (to be filled externally).
        """
        for r in results:
            if not interactive:
                continue
            print("\n" + "="*60)
            print(f"Q: {r.question}")
            print(f"\nExpected: {r.expected_answer[:300]}")
            print(f"\nActual:   {r.actual_answer[:300]}")
            print(f"\nSources:  {r.retrieved_sources}")

            r.human_retrieval_ok = _yn("Retrieval relevant?")
            r.human_faithfulness_ok = _yn("Faithful to context?")
            r.human_completeness_ok = _yn("Answer complete?")
            r.human_no_hallucination = _yn("No hallucination?")
            r.human_notes = input("Notes (Enter to skip): ").strip()

        # Update aggregate human rates in a separate pass if called after filling
        return results

    def print_report(self, report: EvalReport) -> None:
        """Pretty-print a summary of the evaluation report."""
        bar = "=" * 55
        print(f"\n{bar}")
        print(f"  RAG Evaluation Report — {report.run_id}")
        print(bar)
        print(f"  Architecture:         {report.architecture}")
        print(f"  Retrieval strategy:   {report.retrieval_strategy}")
        print(f"  Questions evaluated:  {report.n_questions}")
        print(f"  Timestamp:            {report.timestamp}")
        print(bar)
        print(f"  Avg retrieval relevance: {report.avg_retrieval_relevance:.2%}")
        print(f"  Avg faithfulness:        {report.avg_faithfulness:.2%}")
        print(f"  Exact match rate:        {report.exact_match_rate:.2%}")
        print(f"  Avg BLEU-1:              {report.avg_bleu1:.4f}")
        print(f"  Avg latency (ms):        {report.avg_latency_ms:.1f}")
        if report.human_no_hallucination_rate is not None:
            print(f"  Human no-hallucination:  {report.human_no_hallucination_rate:.2%}")
        print(bar + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yn(prompt: str) -> bool:
    while True:
        ans = input(f"  {prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
