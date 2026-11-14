import unittest
from unittest.mock import AsyncMock, patch

from api_services import (
    NVIDIA_NIM_BASE_URL,
    NVIDIA_PRIMARY_MODEL,
    ApiSettings,
)
from main import ResilientGeneralLlm


class ResilientGeneralLlmTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_system_prompt_and_uses_shared_api_router(self) -> None:
        settings = ApiSettings(
            exa_api_secret=None,
            gemini_api_key="gemini-test",
            metaculus_token="metaculus-test",
            nvidia_api_key="nvidia-test",
            parallel_api_key=None,
            tavily_api_key=None,
        )
        llm = ResilientGeneralLlm(
            provider_model="nvidia_nim",
            temperature=0.2,
            settings=settings,
        )

        with patch(
            "main.complete_llm",
            new=AsyncMock(return_value="Forecast rationale"),
        ) as complete:
            answer = await llm.invoke("Forecast question", system_prompt="Be precise.")

        self.assertEqual(answer, "Forecast rationale")
        call_args = complete.await_args
        self.assertIsNotNone(call_args)
        assert call_args is not None
        self.assertEqual(
            call_args.args[0],
            [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Forecast question"},
            ],
        )
        self.assertEqual(call_args.kwargs["model"], "nvidia_nim")
        self.assertEqual(call_args.kwargs["temperature"], 0.2)
        self.assertEqual(llm.model, f"openai/{NVIDIA_PRIMARY_MODEL}")
        self.assertEqual(llm.litellm_kwargs["api_base"], NVIDIA_NIM_BASE_URL)
