import json
from pathlib import Path
import re

import pytest

from evaluate import (CachedEmbeddings, contains_term, retrieval_metrics, run_eval,
                      score_answer, summarize)
from rag import NOT_FOUND, Answer, Citation, GroundedAnswer, read_pdf
from test_rag import StubModel, TopicEmbeddings, fixture_index

EXAMPLES = Path(__file__).parents[1] / "examples"
EVAL_SETS = [("sample_handbook.pdf", "evaluation_questions.json"),
             ("eval_handbook.pdf", "eval_handbook_questions.json")]
CATEGORIES = {"factual", "paraphrase", "exact_term", "multi_page", "unanswerable"}

LEAVE = {"category": "factual", "question": "How much annual leave?", "expected_pages": [1],
         "must_contain": ["20"], "evidence": []}
WORLD_CUP = {"category": "unanswerable", "question": "Who won the World Cup?", "expected_pages": [],
             "must_contain": [], "evidence": []}


@pytest.mark.parametrize("ranked,expected,result", [
    ([3, 1, 2], [3], {"hit_at_1": True, "hit_at_k": True, "reciprocal_rank": 1.0}),
    ([2, 3, 1], [3], {"hit_at_1": False, "hit_at_k": True, "reciprocal_rank": 0.5}),
    ([2, 5, 3], [3, 5], {"hit_at_1": False, "hit_at_k": True, "reciprocal_rank": 0.5}),
    ([1, 2], [3], {"hit_at_1": False, "hit_at_k": False, "reciprocal_rank": 0.0}),
])
def test_retrieval_metrics_use_first_relevant_rank(ranked, expected, result):
    assert retrieval_metrics(ranked, expected) == result


def test_terms_match_whole_words_and_alternatives():
    assert contains_term("You get 6 sessions.", ["6", "six"])
    assert not contains_term("You get 16 sessions.", "6")
    assert contains_term("Meet in Car Park\nB.", "car park b")
    assert contains_term("Submit LD-3 first.", "ld-3")


def test_score_answerable_question():
    good = Answer("Employees receive 20 days.", [{"page": 1}])
    wrong_page = Answer("Employees receive 20 days.", [{"page": 1}, {"page": 2}])
    assert score_answer(good, LEAVE) == {"refused": False, "answer_correct": True,
                                         "cited_pages": [1], "citation_pages_correct": True}
    assert score_answer(wrong_page, LEAVE)["citation_pages_correct"] is False
    assert score_answer(Answer("Employees receive 24 days.", [{"page": 1}]), LEAVE)["answer_correct"] is False
    refused = score_answer(Answer(NOT_FOUND, []), LEAVE)
    assert refused["refused"] and not refused["answer_correct"]
    assert refused["citation_pages_correct"] is None


def test_score_unanswerable_question():
    assert score_answer(Answer(NOT_FOUND, []), WORLD_CUP)["answer_correct"] is True
    assert score_answer(Answer("Argentina.", [{"page": 1}]), WORLD_CUP)["answer_correct"] is False


def test_summarize_separates_retrieval_refusal_and_hallucination():
    results = [
        {"expected_pages": [1], "hit_at_1": True, "hit_at_k": True, "reciprocal_rank": 1.0,
         "refused": False, "answer_correct": True, "citation_pages_correct": True},
        {"expected_pages": [2], "hit_at_1": False, "hit_at_k": True, "reciprocal_rank": 0.5,
         "refused": True, "answer_correct": False, "citation_pages_correct": None},
        {"expected_pages": [], "refused": False, "answer_correct": False, "citation_pages_correct": None},
    ]
    summary = summarize(results)
    assert summary["mrr"] == 0.75
    assert summary["hit_at_1"] == 0.5
    assert summary["answer_accuracy"] == 0.333
    assert summary["citation_page_accuracy"] == 1.0
    assert summary["false_refusal_rate"] == 0.5
    assert summary["hallucination_rate"] == 1.0


def test_run_eval_through_real_pipeline():
    index, embeddings = fixture_index()
    model = StubModel(GroundedAnswer(supported=True, answer="Employees receive 20 days of annual leave.",
                                     citations=[Citation(source_id=1, quote="20 days of annual leave")]))
    [row] = run_eval([LEAVE], index, embeddings, model)
    assert row["retrieved_pages"][0] == 1
    assert row["reciprocal_rank"] == 1.0
    assert row["answer_correct"] and row["citation_pages_correct"]


def test_retrieval_only_never_calls_model():
    index, embeddings = fixture_index()
    results = run_eval([LEAVE, WORLD_CUP], index, embeddings, model=None)
    assert "answer" not in results[0]
    assert summarize(results)["answer_accuracy"] is None


def test_run_stops_at_first_api_error_and_keeps_results():
    class FailingModel(StubModel):
        def invoke(self, messages):
            raise RuntimeError("quota")

    index, embeddings = fixture_index()
    results = run_eval([LEAVE, WORLD_CUP], index, embeddings, FailingModel(None))
    assert len(results) == 1
    assert "error" in results[0]


def test_ctrl_c_keeps_completed_questions():
    class InterruptOnSecondCall(StubModel):
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return super().invoke(messages)

    index, embeddings = fixture_index()
    model = InterruptOnSecondCall(GroundedAnswer(supported=False, answer="", citations=[]))
    results = run_eval([LEAVE, WORLD_CUP, LEAVE], index, embeddings, model)
    assert len(results) == 1
    assert results[0]["answer"] == NOT_FOUND


def test_cache_sends_each_text_to_api_once(tmp_path):
    class CountingEmbeddings(TopicEmbeddings):
        calls = 0

        def embed_query(self, text):
            CountingEmbeddings.calls += 1
            return super().embed_query(text)

    path = tmp_path / "cache.json"
    first = CachedEmbeddings(CountingEmbeddings(), path)
    vector = first.embed_query("annual leave")
    first.embed_documents(["annual leave"])  # Documents and queries are cached separately.
    second = CachedEmbeddings(CountingEmbeddings(), path)
    assert second.embed_query("annual leave") == vector
    assert CountingEmbeddings.calls == 2
    assert (first.api_calls, second.api_calls) == (2, 0)


@pytest.mark.parametrize("pdf_name,questions_name", EVAL_SETS)
def test_eval_labels_match_the_pdf(pdf_name, questions_name):
    pages = {p.metadata["page"]: re.sub(r"\s+", " ", p.page_content)
             for p in read_pdf((EXAMPLES / pdf_name).read_bytes(), pdf_name).pages}
    for item in json.loads((EXAMPLES / questions_name).read_text()):
        assert item["category"] in CATEGORIES
        assert (item["category"] == "unanswerable") == (not item["expected_pages"])
        if item["expected_pages"]:
            assert item["must_contain"] and item["evidence"]
        for evidence in item["evidence"]:
            assert any(evidence in pages[p] for p in item["expected_pages"]), evidence
