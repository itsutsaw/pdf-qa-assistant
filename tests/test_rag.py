from io import BytesIO
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter

from rag import (Answer, Citation, GroundedAnswer, MAX_BYTES, NOT_FOUND,
                 PDFError, PDFIndex, answer_question, read_pdf, split_pages)

SAMPLE = Path(__file__).parents[1] / "examples" / "sample_handbook.pdf"


class TopicEmbeddings(Embeddings):
    """Deterministic test double; these are NOT production semantic embeddings."""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def close(self):
        pass

    def embed_query(self, text):
        text = text.lower()
        return [float("leave" in text), float("laptop" in text), 0.01]


class StubModel:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.result


def fixture_index():
    documents = [
        Document(page_content="Employees receive 20 days of annual leave.", metadata={"page": 1}),
        Document(page_content="Report a lost laptop to the IT help desk immediately.", metadata={"page": 2}),
    ]
    embeddings = TopicEmbeddings()
    return PDFIndex.build(documents, embeddings), embeddings


def test_extract_and_chunk_preserve_physical_page_numbers():
    parsed = read_pdf(SAMPLE.read_bytes(), SAMPLE.name)
    assert parsed.total_pages == 3
    assert [p.metadata["page"] for p in parsed.pages] == [1, 2, 3]
    assert "20 days" in parsed.pages[0].page_content
    chunks = split_pages(parsed.pages)
    assert all(c.metadata["page"] in (1, 2, 3) for c in chunks)
    assert all(len(c.page_content) <= 1000 for c in chunks)


@pytest.mark.parametrize("data", [b"", b"not a PDF", b"%PDF-1.4\ncorrupt"])
def test_reject_invalid_files(data):
    with pytest.raises(PDFError):
        read_pdf(data, "bad.pdf")


def test_reject_oversized_file():
    with pytest.raises(PDFError, match="10 MB"):
        read_pdf(b"%PDF-" + b" " * MAX_BYTES, "large.pdf")


def test_blank_and_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(PDFError, match="OCR"):
        read_pdf(buffer.getvalue(), "blank.pdf")
    writer.encrypt("password")
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(PDFError, match="Password-protected"):
        read_pdf(buffer.getvalue(), "locked.pdf")


def test_faiss_retrieves_relevant_page_without_padding_results():
    index, embeddings = fixture_index()
    results = index.search("Where do I report a lost laptop?", embeddings, k=4)
    assert len(results) == 2
    assert results[0].metadata["page"] == 2


def test_grounded_answer_returns_verified_quote_and_page():
    index, embeddings = fixture_index()
    model = StubModel(GroundedAnswer(supported=True, answer="You receive 20 days of annual leave.",
                                    citations=[Citation(source_id=1, quote="20 days of annual leave")]))
    answer = answer_question("How much annual leave?", index, embeddings, model)
    assert answer.sources[0]["page"] == 1
    assert "20 days" in answer.text
    assert "untrusted" in model.messages[0].content


@pytest.mark.parametrize("supported,citations", [
    (False, []), (True, []),
    (True, [Citation(source_id=99, quote="20 days")]),
    (True, [Citation(source_id=1, quote="100 days of paid leave")]),
])
def test_unsupported_or_invalid_citations_fail_closed(supported, citations):
    index, embeddings = fixture_index()
    model = StubModel(GroundedAnswer(supported=supported, answer="Unsupported claim", citations=citations))
    assert answer_question("How much leave?", index, embeddings, model) == Answer(NOT_FOUND, [])


@pytest.mark.parametrize("question", ["   ", "x" * 2001])
def test_invalid_questions_are_rejected_before_api_call(question):
    index, embeddings = fixture_index()
    model = StubModel(None)
    with pytest.raises(ValueError):
        answer_question(question, index, embeddings, model)
    assert model.messages is None
