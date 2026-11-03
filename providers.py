"""Gemini Developer API clients; no OpenAI or Vertex AI fallback."""
import json
import re
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
RATE_LIMIT_WAIT = 60.0  # Used when Google does not say how long to wait.
MAX_RATE_LIMIT_WAITS = 2


def api_error(exc: Exception) -> errors.APIError | None:
    """Find Google's error inside wrapper exceptions (LangChain wraps it)."""
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, errors.APIError):
            return current
        current = current.__cause__ or current.__context__
    return None


def is_temporary(exc: Exception) -> bool:
    """True for Google server errors that often succeed on retry (not quota or bad requests)."""
    error = api_error(exc)
    return error is not None and error.code in TEMPORARY_ERROR_CODES


def per_minute_wait(exc: Exception) -> float | None:
    """Seconds to wait if a per-minute limit was hit (it refills within a minute).
    None for daily quota, which waiting cannot fix, and for every other error."""
    error = api_error(exc)
    if error is None or error.code != 429:
        return None
    details = json.dumps(error.details, default=str)
    if "PerDay" in details or "PerMinute" not in details:
        return None
    delay = re.search(r'"retryDelay": "(\d+(?:\.\d+)?)s"', details)
    return min(float(delay.group(1)) + 1, 65.0) if delay else RATE_LIMIT_WAIT


def call_gemini(call, on_wait=None):
    """Run call(), recovering from errors that waiting can fix:
    - per-minute rate limit (429): wait for the quota to refill, at most twice;
    - temporary server error (5xx): retry once after 2 seconds.
    Daily quota and bad-request errors are raised at once; retrying would only spend quota."""
    rate_waits = server_retries = 0
    while True:
        try:
            return call()
        except Exception as exc:
            wait = per_minute_wait(exc)
            if wait is not None and rate_waits < MAX_RATE_LIMIT_WAITS:
                rate_waits += 1
            elif is_temporary(exc) and server_retries < 1:
                server_retries += 1
                wait = RETRY_DELAY
            else:
                raise
        if on_wait:
            on_wait(wait)
        time.sleep(wait)


class GeminiEmbeddings(Embeddings):
    """One vector per passage (Embedding 2 aggregates multiple inputs).
    on_progress(done, total) and on_wait(seconds) let the UI show what is happening."""
    def __init__(self, api_key: str, client=None, model: str = EMBEDDING_MODEL,
                 on_progress=None, on_wait=None):
        self.model = model
        self.on_progress = on_progress
        self.on_wait = on_wait
        self.client = client or genai.Client(
            api_key=api_key, vertexai=False,
            http_options=types.HttpOptions(
                timeout=45000, retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _embed(self, text: str, task_type=None) -> list[float]:
        result = call_gemini(lambda: self.client.models.embed_content(
            model=self.model, contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768, task_type=task_type),
        ), self.on_wait)
        if not result.embeddings or len(result.embeddings) != 1 or not result.embeddings[0].values:
            raise ValueError("Gemini returned an empty embedding. Please try again.")
        return list(result.embeddings[0].values)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for number, text in enumerate(texts, start=1):
            if self.model == "gemini-embedding-001":
                vectors.append(self._embed(text, "RETRIEVAL_DOCUMENT"))
            else:
                vectors.append(self._embed(f"title: none | text: {text}"))
            if self.on_progress:
                self.on_progress(number, len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self.model == "gemini-embedding-001":
            return self._embed(text, "RETRIEVAL_QUERY")
        return self._embed(f"task: question answering | query: {text}")

    def close(self):
        self.client.close()


class RetryingChat(ChatGoogleGenerativeAI):
    """LangChain's own retries would also retry daily quota errors, so retry here instead."""
    def _generate(self, *args, **kwargs):
        return call_gemini(lambda: super(RetryingChat, self)._generate(*args, **kwargs))


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
