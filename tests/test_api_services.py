import os
import unittest
from unittest.mock import Mock, patch

import requests

from api_services import ApiSettings, search_research


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
                "gemini/gemini-2.5-flash",
                "nvidia_nim/meta/llama-3.1-70b-instruct",
            ),
        )
        self.assertEqual(settings.llm_api_key("nvidia_nim/example"), "nvidia-test")

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
        self.assertEqual(settings.llm_models()[0], "gemini/gemini-2.5-flash")


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
        response = Mock()
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

        with patch("api_services.requests.post", return_value=response) as post:
            research = search_research("Forecast question", settings)

        self.assertEqual(post.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in post.call_args_list],
            [
                "https://api.exa.ai/search",
                "https://api.tavily.com/search",
                "https://api.parallel.ai/v1/search",
            ],
        )
        self.assertIn("## Exa", research)
        self.assertIn("## Tavily", research)
        self.assertIn("## Parallel", research)
        self.assertEqual(post.call_args_list[0].kwargs["headers"], {"x-api-key": "exa-test"})
        self.assertEqual(
            post.call_args_list[1].kwargs["json"]["api_key"], "tavily-test"
        )
        self.assertEqual(
            post.call_args_list[2].kwargs["headers"]["x-api-key"],
            "parallel-test",
        )
        self.assertIn("advanced_settings", post.call_args_list[2].kwargs["json"])
        self.assertTrue(all(call.kwargs["timeout"] == 30 for call in post.call_args_list))

    def test_surfaces_provider_http_errors(self) -> None:
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
            with self.assertRaisesRegex(requests.HTTPError, "provider failed"):
                search_research("Forecast question", settings)


if __name__ == "__main__":
    unittest.main()
