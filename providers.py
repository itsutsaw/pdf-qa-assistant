"""Gemini Developer API clients; no OpenAI or Vertex AI fallback."""
import time

import httpx
from google import genai
from google.genai import errors, types
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2"
INDEX_VERSION = f"{EMBEDDING_MODEL}:768:qa-prefix-v1"
TEMPORARY_ERROR_CODES = {500, 502, 503, 504}
RETRY_DELAY = 2.0


def is_temporary(exc: Exception) -> bool:
    """True for Google server errors that often succeed on retry (not quota or bad requests)."""
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, errors.APIError):
            return current.code in TEMPORARY_ERROR_CODES
        current = current.__cause__ or current.__context__
    return False


def retry_once(call):
    """Run call(), retrying one time after a temporary error. Quota errors (429) are never
    retried, because a retry would spend more quota and still fail."""
    try:
        return call()
    except Exception as exc:
        if not is_temporary(exc):
            raise
    time.sleep(RETRY_DELAY)
    return call()


class GeminiEmbeddings(Embeddings):
    """One vector per passage (Embedding 2 aggregates multiple inputs)."""
    def __init__(self, api_key: str, client=None, model: str = EMBEDDING_MODEL):
        self.model = model
        self.client = client or genai.Client(
            api_key=api_key, vertexai=False,
            http_options=types.HttpOptions(
                timeout=45000, retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _embed(self, text: str, task_type=None) -> list[float]:
        result = retry_once(lambda: self.client.models.embed_content(
            model=self.model, contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768, task_type=task_type),
        ))
        if not result.embeddings or len(result.embeddings) != 1 or not result.embeddings[0].values:
            raise ValueError("Gemini returned an empty embedding. Please try again.")
        return list(result.embeddings[0].values)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.model == "gemini-embedding-001":
            return [self._embed(text, "RETRIEVAL_DOCUMENT") for text in texts]
        return [self._embed(f"title: none | text: {text}") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.model == "gemini-embedding-001":
            return self._embed(text, "RETRIEVAL_QUERY")
        return self._embed(f"task: question answering | query: {text}")

    def close(self):
        self.client.close()


class RetryingChat(ChatGoogleGenerativeAI):
    """LangChain's own retries would also retry quota errors, so retry here instead."""
    def _generate(self, *args, **kwargs):
        return retry_once(lambda: super(RetryingChat, self)._generate(*args, **kwargs))


def make_model(api_key: str, model: str = CHAT_MODEL):
    return RetryingChat(
        model=model, google_api_key=api_key, vertexai=False,
        timeout=45, max_retries=0, max_output_tokens=4096,
        # Gemini 3 models use fixed sampling and warn if a temperature is sent.
        **({} if model.startswith("gemini-3") else {"temperature": 0}),
    )


def error_message(exc: Exception) -> str:
    """Unwrap provider errors without displaying credentials or raw responses."""
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, errors.APIError):
            code = current.code
            if code == 429:
                return ("Gemini's request limit or quota was reached. Wait for your quota to reset, "
                        "or try a shorter PDF. Check your project's limits in Google AI Studio; "
                        "you do not need to enable paid billing to keep using the free tier.")
            if code in (400, 401, 403):
                return ("Gemini rejected this request. Check your API key, its permissions, "
                        "and whether the Gemini API is available for your account and region.")
            if code == 404:
                return "This model is unavailable. Click Load available models in the sidebar and select another listed model, then retry."
            return "Gemini is temporarily unavailable. Please try again later."
        if isinstance(current, httpx.TransportError):
            return "Could not reach Gemini. Check your internet connection and try again."
        current = current.__cause__ or current.__context__
    return "The request could not be completed. Please retry or use a smaller PDF."


def list_available_models(api_key: str) -> dict[str, list[str]]:
    """List supported API models; this does not attest to pricing or remaining quota."""
    client = genai.Client(api_key=api_key, vertexai=False,
                         http_options=types.HttpOptions(timeout=20000,
                         retry_options=types.HttpRetryOptions(attempts=1)))
    try:
        chats, embeddings = set(), set()
        for model in client.models.list():
            name = (model.name or "").removeprefix("models/")
            actions = model.supported_actions or []
            if ("generateContent" in actions and name.startswith("gemini-")
                    and "flash" in name and not any(x in name for x in
                    ("image", "audio", "tts", "live", "robotics"))):
                chats.add(name)
            if "embedContent" in actions and name in (
                "gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview"
            ):
                embeddings.add(name)
        preferred = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
        chats = sorted(chats, key=lambda n: (preferred.index(n) if n in preferred else len(preferred), n))
        embeddings = sorted(embeddings, key=lambda n: (n != "gemini-embedding-001", n))
        if not chats or not embeddings:
            raise ValueError("Google did not list both a compatible Flash model and an embedding model. Check this key's project and Gemini API access in AI Studio.")
        return {"chat": chats, "embedding": embeddings}
    finally:
        client.close()
