"""OpenAI chat completions client implementing the LLMClient protocol."""

import os

from openai import OpenAI

from app.llm import LLMReply


class OpenAILLMClient:
    """Calls OpenAI chat completions API. Requires OPENAI_API_KEY in env."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate_reply(self, user_message: str) -> LLMReply:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": user_message}],
        )
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return LLMReply(
            message=content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
