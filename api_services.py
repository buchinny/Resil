from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Any, Callable

import requests


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


@dataclass(frozen=True)
class ApiSettings:
    exa_api_secret: str | None
    gemini_api_key: str | None
    metaculus_token: str | None
    nvidia_api_key: str | None
    parallel_api_key: str | None
    tavily_api_key: str | None

    @classmethod
    def from_env(cls) -> ApiSettings:
        return cls(
            exa_api_secret=_read_secret("EXA_API_SECRET"),
            gemini_api_key=_read_secret("GEMINI_API_KEY"),
            metaculus_token=_read_secret("METACULUS_TOKEN"),
            nvidia_api_key=_read_secret("NVIDIA_API_KEY"),
            parallel_api_key=_read_secret("PARALLEL_API_KEY"),
            tavily_api_key=_read_secret("TAVILY_API_KEY"),
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

    def llm_models(self) -> tuple[str, str]:
        """Return primary and parser models, using both providers when configured."""
        if self.gemini_api_key and self.nvidia_api_key:
            return (
                "gemini/gemini-2.5-flash",
                "nvidia_nim/meta/llama-3.1-70b-instruct",
            )
        if self.gemini_api_key:
            model = "gemini/gemini-2.5-flash"
            return model, model
        if self.nvidia_api_key:
            model = "nvidia_nim/meta/llama-3.1-70b-instruct"
            return model, model
        raise ValueError("No approved LLM API key is configured.")

    def llm_api_key(self, model: str) -> str:
        if model.startswith("gemini/") and self.gemini_api_key:
            return self.gemini_api_key
        if model.startswith("nvidia_nim/") and self.nvidia_api_key:
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
        raise ValueError(f"Expected a JSON object from {url}.")
    return data


def _format_results(provider: str, results: Any) -> str:
    if not isinstance(results, list):
        raise ValueError(f"{provider} search response did not contain a result list.")
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


def search_research(question: str, settings: ApiSettings) -> str:
    providers: list[tuple[str, Callable[[str, str], str], str]] = []
    if settings.exa_api_secret:
        providers.append(("Exa", _exa_search, settings.exa_api_secret))
    if settings.tavily_api_key:
        providers.append(("Tavily", _tavily_search, settings.tavily_api_key))
    if settings.parallel_api_key:
        providers.append(("Parallel", _parallel_search, settings.parallel_api_key))
    if not providers:
        return ""

    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        results = list(
            executor.map(lambda item: item[1](item[2], question), providers)
        )
    return "\n\n".join(results)
