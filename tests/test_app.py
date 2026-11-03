from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from providers import CHAT_MODEL
from rag import Citation, GroundedAnswer
from usage import SESSION_LIMITS
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


DEMO_ANSWER = GroundedAnswer(supported=True, answer="Employees receive 20 days of annual leave.",
                             citations=[Citation(source_id=1, quote="20 days of paid annual leave")])


def demo_app():
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.secrets["GEMINI_API_KEY"] = "demo-key"
    return app.run()


def test_demo_key_needs_no_visitor_setup(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("providers.GeminiEmbeddings", return_value=TopicEmbeddings()), \
         patch("providers.make_model", return_value=StubModel(DEMO_ANSWER)) as make_model, \
         patch("providers.list_available_models") as list_models:
        app = demo_app()
        assert not app.exception
        assert not any(b.label == "Load available models" for b in app.button)
        app.toggle[0].set_value(True).run()
        next(b for b in app.button if b.label == "Prepare PDF").click().run()
        app.chat_input[0].set_value("How much annual leave do employees receive?").run()
        assert not app.exception
        assert "20 days" in app.session_state["messages"][1]["text"]
        assert make_model.call_args.args[0] == "demo-key"
        assert make_model.call_args.kwargs["model"] == CHAT_MODEL
        list_models.assert_not_called()
        assert app.session_state["demo_usage"] == {"pdfs": 1, "questions": 1}


def test_demo_question_limit_blocks_model_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    model = StubModel(DEMO_ANSWER)
    with patch("providers.GeminiEmbeddings", return_value=TopicEmbeddings()), \
         patch("providers.make_model", return_value=model):
        app = demo_app()
        app.toggle[0].set_value(True).run()
        next(b for b in app.button if b.label == "Prepare PDF").click().run()
        app.session_state["demo_usage"] = {"pdfs": 1, "questions": SESSION_LIMITS["questions"]}
        app.chat_input[0].set_value("How much annual leave do employees receive?").run()
        assert not app.exception
        assert model.messages is None
        assert "limit" in app.warning[0].value
        assert app.session_state["messages"] == []


def test_visitor_key_replaces_demo_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = demo_app()
    app.text_input[0].set_value("my-own-key").run()
    assert not app.exception
    assert any(b.label == "Load available models" for b in app.button)
