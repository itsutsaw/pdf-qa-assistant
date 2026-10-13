from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from rag import Citation, GroundedAnswer
from test_rag import TopicEmbeddings, StubModel

APP = Path(__file__).parents[1] / "app.py"


def test_first_run_and_sample_require_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Your PDF, in conversation."
    app.toggle[0].set_value(True).run()
    assert not app.exception
    prepare = next(b for b in app.button if b.label == "Prepare PDF")
    assert prepare.disabled


def test_sample_question_clear_and_document_removal(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = GroundedAnswer(supported=True, answer="Employees receive 20 days of annual leave.",
                              citations=[Citation(source_id=1, quote="20 days of paid annual leave")])
    with patch("providers.GeminiEmbeddings", return_value=TopicEmbeddings()), \
         patch("providers.make_model", return_value=StubModel(response)), \
         patch("providers.list_available_models", return_value={"chat": ["gemini-3.1-flash-lite"], "embedding": ["gemini-embedding-001"]}):
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        app.text_input[0].set_value("test-key").run()
        next(b for b in app.button if b.label == "Load available models").click().run()
        app.toggle[0].set_value(True).run()
        next(b for b in app.button if b.label == "Prepare PDF").click().run()
        assert not app.exception
        assert app.metric[2].value == "Ready"
        app.chat_input[0].set_value("How much annual leave do employees receive?").run()
        assert not app.exception
        assert len(app.chat_message) == 2
        assert "20 days" in app.session_state["messages"][1]["text"]
        assert app.session_state["messages"][1]["sources"][0]["page"] == 1
        next(b for b in app.button if b.label == "Clear conversation").click().run()
        assert app.session_state["messages"] == []
        app.toggle[0].set_value(False).run()
        assert "index" not in app.session_state
        assert len(app.chat_input) == 0
