from __future__ import annotations

import asyncio
import itertools
import logging
import os
import random
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

import requests
from litellm import acompletion
from litellm.exceptions import APIConnectionError, APIError
from litellm.exceptions import Timeout as LiteLLMTimeout

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_MODELS = (
    "deepseek-ai/deepseek-r1",
    "qwen/qwen3-235b-a22b",
    "meta/llama-3.3-70b-instruct",
)
GEMINI_MODEL = "gemini/gemini-2.5-flash"
_NVIDIA_MODEL_CYCLE = itertools.cycle(NVIDIA_NIM_MODELS)
_NVIDIA_MODEL_LOCK = threading.Lock()
_NVIDIA_REQUEST_INTERVAL_SECONDS = 1.5
_NVIDIA_MAX_CONCURRENT_REQUESTS = 2
_NVIDIA_MAX_ATTEMPTS = 3
logger = logging.getLogger(__name__)
T = TypeVar("T")


_PLACEHOLDER_VALUES = {
    "1234567890",
    "REPLACE_ME",
    "your-token-here",
    "your-api-key-here",
}


def _read_secret(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None
    value = value.strip()
    if not value or value in _PLACEHOLDER_VALUES:
        return None
    return value


def _read_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ApiSettings:
    exa_api_secret: str | None
    gemini_api_key: str | None
    metaculus_token: str | None
    nvidia_api_key: str | None
    parallel_api_key: str | None
    tavily_api_key: str | None
    exa_search_enabled: bool = True
    tavily_search_enabled: bool = True
    parallel_search_enabled: bool = True
    gemini_grounded_search_enabled: bool = True

    @classmethod
    def from_env(cls) -> ApiSettings:
        return cls(
            exa_api_secret=_read_secret("EXA_API_SECRET"),
            gemini_api_key=_read_secret("GEMINI_API_KEY"),
            metaculus_token=_read_secret("METACULUS_TOKEN"),
            nvidia_api_key=_read_secret("NVIDIA_API_KEY"),
            parallel_api_key=_read_secret("PARALLEL_API_KEY"),
            tavily_api_key=_read_secret("TAVILY_API_KEY"),
            exa_search_enabled=_read_flag("EXA_SEARCH_ENABLED"),
            tavily_search_enabled=_read_flag("TAVILY_SEARCH_ENABLED"),
            parallel_search_enabled=_read_flag("PARALLEL_SEARCH_ENABLED"),
            gemini_grounded_search_enabled=_read_flag(
                "GEMINI_GROUNDED_SEARCH_ENABLED"
            ),
        )

    def validation_problems(self) -> list[str]:
        problems: list[str] = []
        if not self.metaculus_token:
            problems.append(
                "METACULUS_TOKEN is missing or still a placeholder. "
                "Get one at https://www.metaculus.com/futureeval/participate/"
            )
        if not self.gemini_api_key and not self.nvidia_api_key:
            problems.append(
                "Set at least one LLM key: GEMINI_API_KEY or NVIDIA_API_KEY."
            )
        return problems

    @property
    def has_enabled_search(self) -> bool:
        return any(
            (
                self.exa_api_secret and self.exa_search_enabled,
                self.gemini_api_key and self.gemini_grounded_search_enabled,
                self.parallel_api_key and self.parallel_search_enabled,
                self.tavily_api_key and self.tavily_search_enabled,
            )
        )

    def llm_models(self) -> tuple[str, str]:
        """Return the forecast and parser providers."""
        if self.gemini_api_key and self.nvidia_api_key:
            return "nvidia_nim", GEMINI_MODEL
        if self.gemini_api_key:
            return GEMINI_MODEL, GEMINI_MODEL
        if self.nvidia_api_key:
            return "nvidia_nim", "nvidia_nim"
        raise ValueError("No approved LLM API key is configured.")

    def llm_api_key(self, model: str) -> str:
        if model.startswith("gemini/") and self.gemini_api_key:
            return self.gemini_api_key
        if model == "nvidia_nim" and self.nvidia_api_key:
            return self.nvidia_api_key
        raise ValueError(f"No configured API key for model provider in {model!r}.")


def _post_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object from {url}.")
    return data


def _format_results(provider: str, results: Any) -> str:
    if not isinstance(results, list):
        raise TypeError(f"{provider} search response did not contain a result list.")
    formatted: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        title = result.get("title") or "Untitled result"
        url = result.get("url") or ""
        excerpts: list[str] = []
        for field in ("highlights", "snippets", "excerpts", "content"):
            value = result.get(field)
            if isinstance(value, str) and value:
                excerpts.append(value)
            elif isinstance(value, list):
                excerpts.extend(item for item in value if isinstance(item, str) and item)
        details = "\n".join(excerpts)
        formatted.append(f"### {title}\n{url}\n{details}".strip())
    return f"## {provider}\n" + ("\n\n".join(formatted) or "No results.")


def _exa_search(api_secret: str, question: str) -> str:
    data = _post_json(
        "https://api.exa.ai/search",
        headers={"x-api-key": api_secret},
        payload={
            "query": question,
            "type": "auto",
            "numResults": 6,
            "contents": {"highlights": {"maxCharacters": 1200}},
        },
    )
    return _format_results("Exa", data.get("results"))


def _tavily_search(api_key: str, question: str) -> str:
    data = _post_json(
        "https://api.tavily.com/search",
        payload={
            "api_key": api_key,
            "query": question,
            "search_depth": "advanced",
            "max_results": 6,
            "include_answer": False,
        },
    )
    return _format_results("Tavily", data.get("results"))


def _gemini_grounded_search(api_key: str, question: str) -> str:
    data = _post_json(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent",
        headers={"x-goog-api-key": api_key},
        payload={
            "contents": [{"parts": [{"text": question}]}],
            "tools": [{"google_search": {}}],
        },
    )
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini grounded search returned no candidate.")

    candidate = candidates[0]
    content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
    parts = content.get("parts", []) if isinstance(content, dict) else []
    answer = "\n".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )
    metadata = candidate.get("groundingMetadata", {}) if isinstance(candidate, dict) else {}
    chunks = metadata.get("groundingChunks", []) if isinstance(metadata, dict) else []
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if isinstance(web, dict):
            results.append(
                {
                    "title": web.get("title"),
                    "url": web.get("uri"),
                    "content": answer,
                }
            )
    if not results and answer:
        results.append({"title": "Gemini grounded answer", "content": answer})
    return _format_results("Gemini grounded search", results)


def _parallel_search(api_key: str, question: str) -> str:
    search_query = " ".join(question.split()[:12])
    data = _post_json(
        "https://api.parallel.ai/v1/search",
        headers={"x-api-key": api_key},
        payload={
            "objective": (
                "Find current, reliable information relevant to this forecasting "
                f"question: {question}"
            ),
            "search_queries": [search_query],
            "advanced_settings": {"max_results": 6},
        },
    )
    return _format_results("Parallel", data.get("results"))


class _NvidiaRateLimiter:
    def __init__(self) -> None:
        self._limiters: WeakKeyDictionary[
            asyncio.AbstractEventLoop, tuple[asyncio.Semaphore, asyncio.Lock]
        ] = WeakKeyDictionary()
        self._limiters_lock = threading.Lock()
        self._next_request_at = 0.0

    async def run(self, request: Callable[[], Awaitable[T]]) -> T:
        loop = asyncio.get_running_loop()
        with self._limiters_lock:
            limiter = self._limiters.get(loop)
            if limiter is None:
                limiter = (
                    asyncio.Semaphore(_NVIDIA_MAX_CONCURRENT_REQUESTS),
                    asyncio.Lock(),
                )
                self._limiters[loop] = limiter
        semaphore, spacing_lock = limiter
        async with semaphore:
            async with spacing_lock:
                now = time.monotonic()
                delay = self._next_request_at - now
                if delay > 0:
                    await asyncio.sleep(delay)
                self._next_request_at = (
                    time.monotonic() + _NVIDIA_REQUEST_INTERVAL_SECONDS
                )
            return await request()


_NVIDIA_RATE_LIMITER = _NvidiaRateLimiter()


def _is_temporary_nvidia_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code in {429, 503}:
        return True
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, requests.Timeout)):
        return True
    return type(error).__name__.lower() in {
        "apiconnectionerror",
        "connecttimeout",
        "readtimeout",
        "timeout",
    }


async def _call_nvidia_with_fallback(
    nvidia_call: Callable[[], Awaitable[T]],
    gemini_call: Callable[[], Awaitable[T]] | None,
    *,
    has_gemini_key: bool,
) -> T:
    last_error: Exception | None = None
    for attempt in range(_NVIDIA_MAX_ATTEMPTS):
        try:
            return await _NVIDIA_RATE_LIMITER.run(nvidia_call)
        except (APIConnectionError, APIError, LiteLLMTimeout, TimeoutError, requests.RequestException) as error:
            if not _is_temporary_nvidia_error(error):
                raise
            last_error = error
            if attempt + 1 < _NVIDIA_MAX_ATTEMPTS:
                delay = min(2**attempt, 8) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)

    if has_gemini_key and gemini_call is not None:
        logger.warning(
            "NVIDIA NIM remained unavailable after %s attempts; falling back to Gemini (%s).",
            _NVIDIA_MAX_ATTEMPTS,
            type(last_error).__name__ if last_error else "unknown error",
        )
        return await gemini_call()
    if last_error is not None:
        raise last_error
    raise RuntimeError("NVIDIA NIM request ended without a response.")


def _extract_completion_text(response: Any) -> str:
    try:
        answer = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("LLM response did not contain a completion choice.") from error
    if not isinstance(answer, str) or not answer:
        raise ValueError("No answer returned from LLM.")
    return answer


async def complete_llm(
    prompt: str | list[dict[str, Any]],
    settings: ApiSettings,
    *,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Call Gemini or a free-tier NVIDIA NIM model with throttling and fallback."""
    selected_model = model or settings.llm_models()[0]
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

    async def gemini_call() -> str:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        response = await acompletion(
            model=GEMINI_MODEL,
            messages=messages,
            temperature=temperature,
            api_key=settings.gemini_api_key,
            timeout=60,
        )
        return _extract_completion_text(response)

    if selected_model.startswith("gemini/"):
        return await gemini_call()
    if selected_model != "nvidia_nim" and not selected_model.startswith("nvidia_nim/"):
        raise ValueError(f"Unapproved LLM provider configured: {selected_model!r}.")
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not configured.")

    explicit_model = (
        selected_model.removeprefix("nvidia_nim/")
        if selected_model.startswith("nvidia_nim/")
        else ""
    )
    if explicit_model and explicit_model not in NVIDIA_NIM_MODELS:
        raise ValueError(f"Unapproved NVIDIA NIM model configured: {explicit_model!r}.")
    if explicit_model:
        nim_model = explicit_model
    else:
        with _NVIDIA_MODEL_LOCK:
            nim_model = next(_NVIDIA_MODEL_CYCLE)

    async def nvidia_call() -> str:
        response = await acompletion(
            model=f"openai/{nim_model}",
            messages=messages,
            temperature=temperature,
            api_key=settings.nvidia_api_key,
            base_url=NVIDIA_NIM_BASE_URL,
            timeout=60,
        )
        return _extract_completion_text(response)

    return await _call_nvidia_with_fallback(
        nvidia_call,
        gemini_call if settings.gemini_api_key else None,
        has_gemini_key=bool(settings.gemini_api_key),
    )


def search_research(question: str, settings: ApiSettings) -> str:
    providers: list[tuple[str, Callable[[str, str], str], str]] = []
    if settings.exa_api_secret and settings.exa_search_enabled:
        providers.append(("Exa", _exa_search, settings.exa_api_secret))
    if settings.tavily_api_key and settings.tavily_search_enabled:
        providers.append(("Tavily", _tavily_search, settings.tavily_api_key))
    if settings.parallel_api_key and settings.parallel_search_enabled:
        providers.append(("Parallel", _parallel_search, settings.parallel_api_key))
    if settings.gemini_api_key and settings.gemini_grounded_search_enabled:
        providers.append(
            ("Gemini grounded search", _gemini_grounded_search, settings.gemini_api_key)
        )
    if not providers:
        return ""

    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {
            executor.submit(search, api_key, question): name
            for name, search, api_key in providers
        }
        results = []
        for future, name in futures.items():
            try:
                results.append(future.result())
            except (requests.RequestException, TypeError, ValueError) as error:
                logger.error("%s search failed: %s", name, type(error).__name__)
                results.append(f"## {name}\nSearch failed: {type(error).__name__}.")
    return "\n\n".join(results)
