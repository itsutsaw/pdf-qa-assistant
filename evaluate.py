"""Measure retrieval and answer quality of the real pipeline on a labelled question set.

    export GEMINI_API_KEY=...
    python evaluate.py                    # full run: retrieval + answers
    python evaluate.py --retrieval-only   # no chat model calls

Embeddings are cached in .eval_cache/, so repeat runs only spend chat model calls.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time
import warnings

from langchain_core.embeddings import Embeddings

import providers
from rag import NOT_FOUND, PDFIndex, answer_question, read_pdf, split_pages

CACHE = Path(".eval_cache/embeddings.json")


class CachedEmbeddings(Embeddings):
    """Disk cache around an embedder: each text is sent to the API at most once."""
    def __init__(self, inner: Embeddings, path: Path = CACHE):
        self.inner = inner
        self.name = getattr(inner, "model", type(inner).__name__)
        self.path = Path(path)
        self.vectors = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.api_calls = 0

    def _get(self, kind: str, text: str, embed) -> list[float]:
        key = hashlib.sha256(f"{self.name}|{kind}|{text}".encode()).hexdigest()
        if key not in self.vectors:
            self.vectors[key] = embed(text)
            self.api_calls += 1
            # Save after every new vector so a quota error mid-run keeps earlier work.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.vectors))
        return self.vectors[key]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get("document", t, lambda x: self.inner.embed_documents([x])[0]) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get("query", text, self.inner.embed_query)


def retrieval_metrics(ranked_pages: list[int], expected_pages: list[int]) -> dict:
    """Chunks never cross pages, so a hit is any retrieved chunk on an expected page."""
    rank = next((i for i, page in enumerate(ranked_pages, start=1) if page in expected_pages), None)
    return {"hit_at_1": rank == 1, "hit_at_k": rank is not None,
            "reciprocal_rank": 1 / rank if rank else 0.0}


def contains_term(text: str, term) -> bool:
    """A term is a string or a list of alternatives; matched as whole words, ignoring case."""
    alternatives = [term] if isinstance(term, str) else term
    text = re.sub(r"\s+", " ", text.lower())
    return any(re.search(rf"(?<!\w){re.escape(a.lower())}(?!\w)", text) for a in alternatives)


def score_answer(answer, item: dict) -> dict:
    refused = answer.text == NOT_FOUND
    cited_pages = sorted({source["page"] for source in answer.sources})
    if not item["expected_pages"]:
        return {"refused": refused, "answer_correct": refused, "cited_pages": cited_pages,
                "citation_pages_correct": None}
    correct = not refused and all(contains_term(answer.text, t) for t in item["must_contain"])
    pages_correct = None if refused else bool(cited_pages) and set(cited_pages) <= set(item["expected_pages"])
    return {"refused": refused, "answer_correct": correct, "cited_pages": cited_pages,
            "citation_pages_correct": pages_correct}


def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[dict]) -> dict:
    answerable = [r for r in results if r["expected_pages"]]
    unanswerable = [r for r in results if not r["expected_pages"]]
    answered = [r for r in results if "answer_correct" in r]
    return {
        "questions": len(results),
        "hit_at_1": mean(r["hit_at_1"] for r in answerable),
        "hit_at_k": mean(r["hit_at_k"] for r in answerable),
        "mrr": mean(r["reciprocal_rank"] for r in answerable),
        "answer_accuracy": mean(r["answer_correct"] for r in answered),
        # Of answers that were given, how often every cited page was an expected page.
        "citation_page_accuracy": mean(r.get("citation_pages_correct") for r in answered),
        "false_refusal_rate": mean(r["refused"] for r in answerable if "refused" in r),
        "hallucination_rate": mean(not r["refused"] for r in unanswerable if "refused" in r),
    }


def run_eval(questions: list[dict], index: PDFIndex, embeddings: Embeddings, model=None,
             k: int = 4, delay: float = 0.0) -> list[dict]:
    """Pass model=None for retrieval only. Stops at the first API error or Ctrl+C and keeps
    the results gathered so far."""
    results = []
    try:
        for number, item in enumerate(questions, start=1):
            if model is not None:
                print(f"[{number}/{len(questions)}] {item['question']}", flush=True)
            ranked_pages = [d.metadata["page"] for d in index.search(item["question"], embeddings, k=k)]
            row = {key: item[key] for key in ("category", "question", "expected_pages")}
            row["retrieved_pages"] = ranked_pages
            if item["expected_pages"]:
                row.update(retrieval_metrics(ranked_pages, item["expected_pages"]))
            results.append(row)
            if model is None:
                continue
            if number > 1:
                time.sleep(delay)  # Stay under free-tier requests-per-minute limits.
            try:
                answer = answer_question(item["question"], index, embeddings, model)
            except Exception as exc:
                row["error"] = providers.error_message(exc)
                print(f"Stopped at question {number}: {row['error']}")
                break
            row["answer"] = answer.text
            row.update(score_answer(answer, item))
    except KeyboardInterrupt:
        # Drop a question interrupted before its answer was scored.
        if results and model is not None and "answer" not in results[-1]:
            results.pop()
        print(f"\nInterrupted; keeping {len(results)} completed questions.")
    return results


def summary_table(results: list[dict]) -> str:
    groups = {"all": results}
    for row in results:
        groups.setdefault(row["category"], []).append(row)
    columns = {"questions": "n", "hit_at_1": "hit@1", "hit_at_k": "hit@k", "mrr": "mrr",
               "answer_accuracy": "answer_acc", "citation_page_accuracy": "cite_page_acc",
               "false_refusal_rate": "false_refuse", "hallucination_rate": "hallucinate"}
    lines = [f"{'category':<14}" + "".join(f"{label:>14}" for label in columns.values())]
    for name, rows in groups.items():
        summary = summarize(rows)
        lines.append(f"{name:<14}" + "".join(
            f"{'-' if summary[c] is None else summary[c]:>14}" for c in columns))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", default="examples/eval_handbook.pdf")
    parser.add_argument("--questions", default="examples/eval_handbook_questions.json")
    parser.add_argument("--k", type=int, default=4, help="Passages for retrieval metrics (answers use the app's k=4).")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip chat model calls.")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between chat model calls.")
    parser.add_argument("--chat-model", default=providers.CHAT_MODEL)
    parser.add_argument("--embedding-model", default=providers.EMBEDDING_MODEL)
    parser.add_argument("--output", help="Results JSON path (default: results/eval-<time>.json).")
    args = parser.parse_args()
    # Gemini 3 models ignore temperature; LangChain warns on every call.
    warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in your environment first.")
    pdf = Path(args.pdf)
    questions = json.loads(Path(args.questions).read_text())
    gemini = providers.GeminiEmbeddings(api_key, model=args.embedding_model)
    embeddings = CachedEmbeddings(gemini)
    try:
        index = PDFIndex.build(split_pages(read_pdf(pdf.read_bytes(), pdf.name).pages), embeddings)
        model = None if args.retrieval_only else providers.make_model(api_key, args.chat_model)
        results = run_eval(questions, index, embeddings, model, k=args.k, delay=args.delay)
    except Exception as exc:
        raise SystemExit(providers.error_message(exc)) from exc
    finally:
        gemini.close()

    print(summary_table(results))
    print(f"\nEmbedding API calls this run: {embeddings.api_calls}")
    output = Path(args.output or f"results/eval-{datetime.now():%Y%m%d-%H%M%S}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "config": {"pdf": str(pdf), "questions": args.questions, "k": args.k,
                   "chat_model": None if args.retrieval_only else args.chat_model,
                   "embedding_model": args.embedding_model, "date": datetime.now().isoformat()},
        "summary": summarize(results),
        "results": results,
    }, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
