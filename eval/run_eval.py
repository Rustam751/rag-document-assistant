"""Evaluation harness for the RAG pipeline.

Measures, per question:
  - retrieval hit: was the expected source (and page, if given) in the top-k results?
  - answer keyword coverage: fraction of expected keywords present in the answer
  - groundedness: did the model return citations (or correctly abstain)?

Usage:
    python eval/run_eval.py [--questions eval/questions.json] [--k 5]

Requires ingested documents and ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

from rag_assistant.pipeline import RAGPipeline

# Phrases that indicate the model (correctly) declined to state an unsupported fact,
# even when it cited sources for the parts it *could* answer.
ABSTAIN_MARKERS = (
    "do not contain",
    "does not contain",
    "do not state",
    "does not state",
    "do not specify",
    "does not specify",
    "do not report",
    "does not report",
    "do not indicate",
    "does not indicate",
    "not measured",
    "not reported",
    "not enough information",
    "no information",
)


def normalize(text: str) -> str:
    """Casefold + strip diacritics so keyword matching survives Unicode quirks
    (e.g. Turkish dotted/dotless I: 'Işık' vs 'IŞIK' must match)."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return stripped.replace("ı", "i")  # dotless ı → i


def evaluate(questions_path: Path, k: int) -> dict:
    spec = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = [q for q in spec["questions"] if not q["question"].startswith("REPLACE ME")]
    if not questions:
        raise SystemExit(
            "eval/questions.json still contains only placeholder questions. "
            "Fill it in for your ingested documents first."
        )

    pipeline = RAGPipeline()
    results = []
    for q in questions:
        start = time.perf_counter()
        result = pipeline.ask(q["question"], k=k)
        latency = time.perf_counter() - start

        retrieved_pairs = {(r.source, r.page) for r in result.retrieved}
        retrieved_sources = {r.source for r in result.retrieved}
        if q["expected_source"] is None:
            # Unanswerable question: correct behavior is abstaining outright OR
            # explicitly qualifying that the sources don't state the asked fact.
            retrieval_hit = None
            abstained = (
                not result.grounded
                or not result.citations
                or any(marker in result.answer.lower() for marker in ABSTAIN_MARKERS)
            )
        else:
            abstained = None
            if q["expected_page"] is not None:
                retrieval_hit = (q["expected_source"], q["expected_page"]) in retrieved_pairs
            else:
                retrieval_hit = q["expected_source"] in retrieved_sources

        answer_norm = normalize(result.answer)
        keywords = q.get("expected_keywords", [])
        matched = [kw for kw in keywords if normalize(kw) in answer_norm]
        coverage = len(matched) / len(keywords) if keywords else 1.0

        results.append(
            {
                "question": q["question"],
                "retrieval_hit": retrieval_hit,
                "abstained_correctly": abstained,
                "keyword_coverage": round(coverage, 2),
                "grounded": result.grounded,
                "n_citations": len(result.citations),
                "latency_s": round(latency, 2),
                "answer": result.answer,
            }
        )

    answerable = [r for r in results if r["retrieval_hit"] is not None]
    unanswerable = [r for r in results if r["abstained_correctly"] is not None]
    summary = {
        "n_questions": len(results),
        "retrieval_hit_rate": (
            round(sum(r["retrieval_hit"] for r in answerable) / len(answerable), 3)
            if answerable
            else None
        ),
        "abstention_rate_on_unanswerable": (
            round(sum(r["abstained_correctly"] for r in unanswerable) / len(unanswerable), 3)
            if unanswerable
            else None
        ),
        "mean_keyword_coverage": round(
            sum(r["keyword_coverage"] for r in results) / len(results), 3
        ),
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 2),
    }
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    report = evaluate(args.questions, args.k)

    out_path = Path("eval/results.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Evaluation summary ===")
    for key, value in report["summary"].items():
        print(f"{key:38s} {value}")
    print(f"\nPer-question results written to {out_path}")


if __name__ == "__main__":
    main()
