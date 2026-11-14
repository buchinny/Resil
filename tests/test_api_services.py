import os
import unittest
from unittest.mock import Mock, patch

import requests
from litellm.exceptions import APIError, RateLimitError

import api_services
from api_services import (
    GEMINI_MODEL,
    NVIDIA_NIM_BASE_URL,
    NVIDIA_NIM_MODELS,
    NVIDIA_PRIMARY_MODEL,
    NVIDIA_SECONDARY_MODELS,
    NVIDIA_SUMMARIZER_MODELS,
    ApiSettings,
    complete_llm,
    normalize_nvidia_model_id,
    refresh_nvidia_model_catalog,
    search_research,
)


def _completion(text: str) -> Mock:
    response = Mock()
    response.choices = [Mock(message=Mock(content=text))]
    return response


class ApiSettingsTests(unittest.TestCase):
    def test_reads_all_allowlisted_keys_and_selects_both_llms(self) -> None:
        values = {
            "EXA_API_SECRET": "exa-test",
            "GEMINI_API_KEY": "gemini-test",
            "METACULUS_TOKEN": "metaculus-test",
            "NVIDIA_API_KEY": "nvidia-test",
            "PARALLEL_API_KEY": "parallel-test",
            "TAVILY_API_KEY": "tavily-test",
        }
        with patch.dict(os.environ, values, clear=True):
            settings = ApiSettings.from_env()

        self.assertEqual(settings.validation_problems(), [])
        self.assertEqual(
            settings.llm_models(),
            (
                f"nvidia_nim/{NVIDIA_PRIMARY_MODEL}",
                f"nvidia_nim/{NVIDIA_SUMMARIZER_MODELS[0]}",
            ),
        )
        self.assertEqual(
            settings.llm_api_key(f"nvidia_nim/{NVIDIA_PRIMARY_MODEL}"),
            "nvidia-test",
        )

    def test_requires_metaculus_and_at_least_one_llm_but_not_search_keys(self) -> None:
        placeholders = {
            "METACULUS_TOKEN": "REPLACE_ME",
            "GEMINI_API_KEY": "REPLACE_ME",
            "NVIDIA_API_KEY": "REPLACE_ME",
        }
        with patch.dict(os.environ, placeholders, clear=True):
            settings = ApiSettings.from_env()

        self.assertEqual(len(settings.validation_problems()), 2)

        with patch.dict(
            os.environ,
            {"METACULUS_TOKEN": "metaculus-test", "GEMINI_API_KEY": "gemini-test"},
            clear=True,
        ):
            settings = ApiSettings.from_env()

        self.assertEqual(settings.validation_problems(), [])
        self.assertIsNone(settings.exa_api_secret)
        self.assertEqual(settings.llm_models()[0], GEMINI_MODEL)

    def test_reads_search_source_flags(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EXA_SEARCH_ENABLED": "false",
                "GEMINI_GROUNDED_SEARCH_ENABLED": "off",
                "PARALLEL_SEARCH_ENABLED": "0",
                "TAVILY_SEARCH_ENABLED": "true",
            },
            clear=True,
        ):
            settings = ApiSettings.from_env()

        self.assertFalse(settings.exa_search_enabled)
        self.assertFalse(settings.gemini_grounded_search_enabled)
        self.assertFalse(settings.parallel_search_enabled)
        self.assertTrue(settings.tavily_search_enabled)

    def test_reads_catalog_check_flag_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env()
        self.assertFalse(settings.nvidia_catalog_check_enabled)

    def test_disables_search_providers_without_making_requests(self) -> None:
        settings = ApiSettings(
            exa_api_secret="exa-test",
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key=None,
            parallel_api_key="parallel-test",
            tavily_api_key="tavily-test",
            exa_search_enabled=False,
            tavily_search_enabled=False,
            parallel_search_enabled=False,
            gemini_grounded_search_enabled=False,
        )
        with patch("api_services.requests.post") as post:
            research = search_research("Forecast question", settings)

        self.assertEqual(research, "")
        self.assertFalse(settings.has_enabled_search)
        post.assert_not_called()


class SearchResearchTests(unittest.TestCase):
    def test_queries_every_configured_search_provider(self) -> None:
        settings = ApiSettings(
            exa_api_secret="exa-test",
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key="parallel-test",
            tavily_api_key="tavily-test",
        )

        def response_for(url: str, **kwargs: object) -> Mock:
            response = Mock()
            if "generateContent" in url:
                response.json.return_value = {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "Grounded detail"}]},
                            "groundingMetadata": {
                                "groundingChunks": [
                                    {
                                        "web": {
                                            "title": "Grounded source",
                                            "uri": "https://grounded.example",
                                        }
                                    }
                                ]
                            },
                        }
                    ]
                }
            else:
                response.json.return_value = {
                    "results": [
                        {
                            "title": "Source",
                            "url": "https://example.test",
                            "highlights": ["Relevant detail"],
                            "content": "Relevant detail",
                        }
                    ]
                }
            return response

        with patch("api_services.requests.post", side_effect=response_for) as post:
            research = search_research("Forecast question", settings)

        self.assertEqual(post.call_count, 4)
        calls_by_url = {call.args[0]: call for call in post.call_args_list}
        self.assertEqual(
            set(calls_by_url),
            {
                "https://api.exa.ai/search",
                "https://api.tavily.com/search",
                "https://api.parallel.ai/v1/search",
                (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    "gemini-2.5-flash:generateContent"
                ),
            },
        )
        self.assertIn("## Exa", research)
        self.assertIn("## Tavily", research)
        self.assertIn("## Parallel", research)
        self.assertIn("## Gemini grounded search", research)
        self.assertEqual(
            calls_by_url["https://api.exa.ai/search"].kwargs["headers"],
            {"x-api-key": "exa-test"},
        )
        self.assertEqual(
            calls_by_url["https://api.tavily.com/search"].kwargs["json"]["api_key"],
            "tavily-test",
        )
        self.assertEqual(
            calls_by_url["https://api.parallel.ai/v1/search"].kwargs["headers"]["x-api-key"],
            "parallel-test",
        )
        self.assertIn(
            "advanced_settings",
            calls_by_url["https://api.parallel.ai/v1/search"].kwargs["json"],
        )
        grounded_call = calls_by_url[
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent"
        ]
        self.assertEqual(grounded_call.kwargs["headers"], {"x-goog-api-key": "gemini-test"})
        self.assertEqual(
            grounded_call.kwargs["json"]["tools"], [{"google_search": {}}]
        )
        self.assertTrue(all(call.kwargs["timeout"] == 30 for call in post.call_args_list))

    def test_reports_provider_http_errors_without_losing_other_results(self) -> None:
        settings = ApiSettings(
            exa_api_secret="exa-test",
            gemini_api_key=None,
            metaculus_token="metaculus-test",
            nvidia_api_key=None,
            parallel_api_key=None,
            tavily_api_key=None,
        )
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("provider failed")

        with patch("api_services.requests.post", return_value=response):
            research = search_research("Forecast question", settings)

        self.assertIn("## Exa", research)
        self.assertIn("Search failed: HTTPError", research)


class NvidiaFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_nim_primary_uses_free_endpoint_and_normalized_model_id(self) -> None:
        settings = ApiSettings(
            exa_api_secret=None,
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key=None,
            tavily_api_key=None,
        )
        call_args: list[dict[str, object]] = []

        async def fake_completion(**kwargs: object) -> Mock:
            call_args.append(kwargs)
            return _completion("NIM response")

        async def immediate_run(request):
            return await request()

        with (
            patch("api_services.acompletion", side_effect=fake_completion),
            patch.object(
                api_services._NVIDIA_RATE_LIMITER,
                "run",
                new=immediate_run,
            ),
        ):
            answer = await complete_llm(
                "Forecast prompt",
                settings,
                model=f"nvidia_nim/{NVIDIA_PRIMARY_MODEL}",
            )

        self.assertEqual(answer, "NIM response")
        self.assertEqual(len(call_args), 1)
        self.assertEqual(call_args[0]["model"], f"openai/{NVIDIA_PRIMARY_MODEL}")
        self.assertEqual(call_args[0]["api_base"], NVIDIA_NIM_BASE_URL)
        self.assertNotIn("base_url", call_args[0])

    def test_strips_repeated_litellm_prefixes(self) -> None:
        self.assertEqual(
            normalize_nvidia_model_id(
                f"openai/nvidia_nim/openai/{NVIDIA_PRIMARY_MODEL}"
            ),
            NVIDIA_PRIMARY_MODEL,
        )

    async def test_transient_errors_fall_through_roster_then_gemini(self) -> None:
        settings = ApiSettings(
            exa_api_secret=None,
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key=None,
            tavily_api_key=None,
        )

        responses = [
            RateLimitError("throttled", "openai", model)
            for model in NVIDIA_NIM_MODELS
        ] + [
            _completion("Gemini fallback")
        ]

        async def fake_completion(**kwargs: object) -> Mock:
            call_args.append(kwargs)
            value = responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        async def immediate_run(request):
            return await request()

        call_args: list[dict[str, object]] = []
        with (
            patch("api_services.acompletion", side_effect=fake_completion),
            patch.object(
                api_services._NVIDIA_RATE_LIMITER,
                "run",
                new=immediate_run,
            ),
        ):
            answer = await complete_llm(
                "Forecast prompt",
                settings,
                model=f"nvidia_nim/{NVIDIA_PRIMARY_MODEL}",
            )

        self.assertEqual(answer, "Gemini fallback")
        self.assertEqual(len(call_args), len(NVIDIA_NIM_MODELS) + 1)
        self.assertTrue(
            [call["model"] for call in call_args[:-1]]
            == [f"openai/{model}" for model in NVIDIA_NIM_MODELS]
        )
        self.assertTrue(
            all(call["api_base"] == NVIDIA_NIM_BASE_URL for call in call_args[:-1])
        )
        self.assertEqual(call_args[-1]["model"], GEMINI_MODEL)
        self.assertEqual(call_args[-1]["api_key"], "gemini-test")

    async def test_410_marks_model_dead_and_skips_it_on_later_calls(self) -> None:
        settings = ApiSettings(
            exa_api_secret=None,
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key=None,
            tavily_api_key=None,
        )
        call_args: list[dict[str, object]] = []
        eol = APIError(410, "model gone", "openai", NVIDIA_PRIMARY_MODEL)

        async def fake_completion(**kwargs: object) -> Mock:
            call_args.append(kwargs)
            if len(call_args) == 1:
                raise eol
            return _completion("NIM response")

        async def immediate_run(request):
            return await request()

        with (
            patch("api_services.acompletion", side_effect=fake_completion),
            patch.object(api_services, "_DEAD_MODELS", set()),
            patch.object(
                api_services._NVIDIA_RATE_LIMITER,
                "run",
                new=immediate_run,
            ),
        ):
            first = await complete_llm(
                "Forecast prompt",
                settings,
                model=f"nvidia_nim/{NVIDIA_PRIMARY_MODEL}",
            )
            after_first = len(call_args)
            second = await complete_llm(
                "Another prompt",
                settings,
                model=f"nvidia_nim/{NVIDIA_PRIMARY_MODEL}",
            )

        self.assertEqual(first, "NIM response")
        self.assertEqual(second, "NIM response")
        self.assertEqual(call_args[0]["model"], f"openai/{NVIDIA_PRIMARY_MODEL}")
        self.assertEqual(
            call_args[1]["model"],
            f"openai/{NVIDIA_SECONDARY_MODELS[0]}",
        )
        self.assertEqual(
            call_args[after_first]["model"],
            f"openai/{NVIDIA_SECONDARY_MODELS[0]}",
        )

    async def test_404_model_is_cached_and_catalog_filters_candidates(self) -> None:
        settings = ApiSettings(
            exa_api_secret=None,
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key=None,
            tavily_api_key=None,
        )
        call_args: list[dict[str, object]] = []
        not_found = api_services.NotFoundError(
            "model not found",
            NVIDIA_PRIMARY_MODEL,
            "openai",
        )
        response = Mock()
        response.json.return_value = {
            "data": [
                {"id": NVIDIA_PRIMARY_MODEL},
                {"id": NVIDIA_SECONDARY_MODELS[1]},
            ]
        }

        async def fake_completion(**kwargs: object) -> Mock:
            call_args.append(kwargs)
            if len(call_args) == 1:
                raise not_found
            return _completion("NIM response")

        async def immediate_run(request):
            return await request()

        with (
            patch("api_services.requests.get", return_value=response),
            patch("api_services.acompletion", side_effect=fake_completion),
            patch.object(api_services, "_DEAD_MODELS", set()),
            patch.object(api_services, "_available_nvidia_models", None),
            patch.object(
                api_services._NVIDIA_RATE_LIMITER,
                "run",
                new=immediate_run,
            ),
        ):
            available = refresh_nvidia_model_catalog(settings)
            answer = await complete_llm(
                "Forecast prompt",
                settings,
                model=f"nvidia_nim/{NVIDIA_PRIMARY_MODEL}",
            )

        self.assertEqual(
            available,
            (NVIDIA_PRIMARY_MODEL, NVIDIA_SECONDARY_MODELS[1]),
        )
        self.assertEqual(answer, "NIM response")
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0]["model"], f"openai/{NVIDIA_PRIMARY_MODEL}")
        self.assertEqual(
            call_args[1]["model"],
            f"openai/{NVIDIA_SECONDARY_MODELS[1]}",
        )

    def test_filters_roster_from_live_nvidia_catalog(self) -> None:
        settings = ApiSettings(
            exa_api_secret=None,
            gemini_api_key=None,
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key=None,
            tavily_api_key=None,
        )
        response = Mock()
        response.json.return_value = {
            "data": [{"id": NVIDIA_SECONDARY_MODELS[1]}, {"id": "unconfigured/model"}]
        }
        with (
            patch("api_services.requests.get", return_value=response) as get,
            patch.object(api_services, "_available_nvidia_models", None),
        ):
            available = refresh_nvidia_model_catalog(settings)

        self.assertEqual(available, (NVIDIA_SECONDARY_MODELS[1],))
        get.assert_called_once_with(
            f"{NVIDIA_NIM_BASE_URL}/models",
            headers={"Authorization": "Bearer nvidia-test"},
            timeout=15,
        )

    async def test_rejects_models_outside_the_approved_nim_roster(self) -> None:
        settings = ApiSettings(
            exa_api_secret=None,
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key=None,
            tavily_api_key=None,
        )
        with self.assertRaisesRegex(ValueError, "Unapproved NVIDIA NIM model"):
            await complete_llm(
                "Forecast prompt",
                settings,
                model="nvidia_nim/unapproved/model",
            )

    async def test_does_not_fallback_for_non_transient_nim_errors(self) -> None:
        settings = ApiSettings(
            exa_api_secret=None,
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key=None,
            tavily_api_key=None,
        )

        unauthorized = requests.HTTPError("denied")
        unauthorized.response = Mock(status_code=401)

        async def immediate_run(request):
            return await request()

        with (
            patch("api_services.acompletion", side_effect=unauthorized) as call,
            patch.object(
                api_services._NVIDIA_RATE_LIMITER,
                "run",
                new=immediate_run,
            ),
            self.assertRaisesRegex(requests.HTTPError, "denied"),
        ):
            await complete_llm("Forecast prompt", settings, model="nvidia_nim")

        call.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
