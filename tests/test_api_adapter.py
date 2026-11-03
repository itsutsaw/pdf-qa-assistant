"""Exercise real Gemini adapters with offline SDK responses; no paid API calls."""
import json
from unittest.mock import patch

import pytest
from google.genai import errors, types

from providers import CHAT_MODEL, EMBEDDING_MODEL, GeminiEmbeddings, error_message, make_model
from rag import answer_question
from test_rag import fixture_index


def test_actual_chat_adapter_parses_structured_response():
    content = {"supported": True, "answer": "20 days of annual leave.",
               "citations": [{"source_id": 1, "quote": "20 days of annual leave"}]}
    response = types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(content)}]},
                        "finish_reason": "STOP"}],
        "usage_metadata": {"prompt_token_count": 30, "candidates_token_count": 20, "total_token_count": 50},
    })
    model = make_model("offline-test-key")
    with patch.object(model.client.models, "generate_content", return_value=response) as generate:
        index, embeddings = fixture_index()
        answer = answer_question("How much leave?", index, embeddings, model)
        assert answer.sources[0]["page"] == 1
        assert generate.call_args.kwargs["model"] == CHAT_MODEL
        config = generate.call_args.kwargs["config"]
        assert config.response_mime_type == "application/json"
    model.client.close()


def test_embedding_adapter_uses_one_request_per_chunk_and_query_prefix():
    embeddings = GeminiEmbeddings("offline-test-key")
    response = types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[1.0, 0.0])])
    with patch.object(embeddings.client.models, "embed_content", return_value=response) as embed:
        assert embeddings.embed_documents(["First passage.", "Second passage."]) == [[1.0, 0.0], [1.0, 0.0]]
        assert embeddings.embed_query("A question?") == [1.0, 0.0]
        assert embed.call_count == 3
        calls = embed.call_args_list
        assert calls[0].kwargs["contents"] == "title: none | text: First passage."
        assert calls[1].kwargs["contents"] == "title: none | text: Second passage."
        assert calls[2].kwargs["contents"] == "task: question answering | query: A question?"
        assert calls[0].kwargs["model"] == EMBEDDING_MODEL
        assert calls[0].kwargs["config"].task_type is None
        assert calls[0].kwargs["config"].output_dimensionality == 768
    embeddings.close()


@pytest.mark.parametrize("code,expected", [(429, "quota"), (403, "permissions"), (404, "unavailable"), (500, "temporarily")])
def test_wrapped_google_errors_provide_actionable_messages(code, expected):
    cause = errors.APIError(code, {"error": {"message": "sensitive raw response"}})
    wrapped = RuntimeError("Provider wrapper")
    wrapped.__cause__ = cause
    message = error_message(wrapped)
    assert expected in message
    assert "sensitive raw response" not in message


def test_model_discovery_filters_capabilities_and_prefers_listed_models():
    from unittest.mock import MagicMock
    from providers import list_available_models
    client = MagicMock()
    client.models.list.return_value = [
        types.Model(name="models/gemini-3.1-flash-lite", supported_actions=["generateContent"]),
        types.Model(name="models/gemini-embedding-001", supported_actions=["embedContent"]),
        types.Model(name="models/gemini-embedding-2", supported_actions=["embedContent"]),
        types.Model(name="models/gemini-flash-image", supported_actions=["generateContent"]),
        types.Model(name="models/gemini-2.5-flash-lite", supported_actions=[]),
    ]
    with patch("providers.genai.Client", return_value=client):
        models = list_available_models("test-key")
    assert models["chat"] == ["gemini-3.1-flash-lite"]
    assert models["embedding"][0] == "gemini-embedding-001"
    client.close.assert_called_once()


def test_embedding_001_uses_retrieval_task_types_without_prefixes():
    embeddings = GeminiEmbeddings("offline-test-key", model="gemini-embedding-001")
    response = types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[1.0, 0.0])])
    with patch.object(embeddings.client.models, "embed_content", return_value=response) as embed:
        embeddings.embed_documents(["A passage"])
        embeddings.embed_query("A question")
        assert embed.call_args_list[0].kwargs["contents"] == "A passage"
        assert embed.call_args_list[0].kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"
        assert embed.call_args_list[1].kwargs["config"].task_type == "RETRIEVAL_QUERY"
    embeddings.close()


def chat_response(content):
    return types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(content)}]},
                        "finish_reason": "STOP"}],
    })


def test_chat_retries_once_after_temporary_server_error(monkeypatch):
    monkeypatch.setattr("providers.RETRY_DELAY", 0)
    content = {"supported": True, "answer": "20 days of annual leave.",
               "citations": [{"source_id": 1, "quote": "20 days of annual leave"}]}
    model = make_model("offline-test-key")
    overloaded = errors.ServerError(503, {"error": {"message": "overloaded"}})
    with patch.object(model.client.models, "generate_content",
                      side_effect=[overloaded, chat_response(content)]) as generate:
        index, embeddings = fixture_index()
        assert answer_question("How much leave?", index, embeddings, model).sources[0]["page"] == 1
        assert generate.call_count == 2
    model.client.close()


@pytest.mark.parametrize("code", [429, 400])
def test_quota_and_request_errors_are_not_retried(monkeypatch, code):
    monkeypatch.setattr("providers.RETRY_DELAY", 0)
    embeddings = GeminiEmbeddings("offline-test-key")
    error = errors.APIError(code, {"error": {"message": "no"}})
    with patch.object(embeddings.client.models, "embed_content", side_effect=error) as embed:
        with pytest.raises(errors.APIError):
            embeddings.embed_query("A question?")
        assert embed.call_count == 1
    embeddings.close()


def test_embedding_gives_up_after_one_retry(monkeypatch):
    monkeypatch.setattr("providers.RETRY_DELAY", 0)
    embeddings = GeminiEmbeddings("offline-test-key")
    error = errors.ServerError(503, {"error": {"message": "overloaded"}})
    with patch.object(embeddings.client.models, "embed_content", side_effect=error) as embed:
        with pytest.raises(errors.ServerError):
            embeddings.embed_query("A question?")
        assert embed.call_count == 2
    embeddings.close()


def quota_error(quota_id, retry_delay="3s"):
    """Shaped like Google's 429 body: which quota was exceeded, and when to retry."""
    return errors.ClientError(429, {"error": {
        "code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ]}})


PER_MINUTE = "EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY = "EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"
VECTOR = types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[1.0, 0.0])])


def test_per_minute_limit_waits_then_resumes_with_progress(monkeypatch):
    sleeps, waits, progress = [], [], []
    monkeypatch.setattr("providers.time.sleep", sleeps.append)
    embeddings = GeminiEmbeddings("offline-test-key", on_wait=waits.append,
                                  on_progress=lambda done, total: progress.append((done, total)))
    with patch.object(embeddings.client.models, "embed_content",
                      side_effect=[VECTOR, quota_error(PER_MINUTE), VECTOR]) as embed:
        assert len(embeddings.embed_documents(["First.", "Second."])) == 2
        assert embed.call_count == 3
    assert sleeps == waits == [4.0]  # Google's retryDelay plus one second.
    assert progress == [(1, 2), (2, 2)]
    embeddings.close()


def test_daily_quota_is_never_waited_for(monkeypatch):
    sleeps = []
    monkeypatch.setattr("providers.time.sleep", sleeps.append)
    embeddings = GeminiEmbeddings("offline-test-key")
    with patch.object(embeddings.client.models, "embed_content", side_effect=quota_error(PER_DAY)) as embed:
        with pytest.raises(errors.ClientError):
            embeddings.embed_query("A question?")
        assert embed.call_count == 1
    assert sleeps == []
    embeddings.close()


def test_per_minute_limit_gives_up_after_two_waits(monkeypatch):
    monkeypatch.setattr("providers.time.sleep", lambda seconds: None)
    embeddings = GeminiEmbeddings("offline-test-key")
    with patch.object(embeddings.client.models, "embed_content", side_effect=quota_error(PER_MINUTE)) as embed:
        with pytest.raises(errors.ClientError):
            embeddings.embed_query("A question?")
        assert embed.call_count == 3
    embeddings.close()
