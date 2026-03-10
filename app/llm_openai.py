"""OpenAI chat completions client implementing the LLMClient protocol."""

import os
import uuid

from openai import OpenAI

from app.llm import LLMReply


class OpenAILLMClient:
    """Calls OpenAI chat completions API. Requires OPENAI_API_KEY in env."""

    # Shared in-memory transcript storage across client instances in-process.
    _sessions: dict[str, list[dict[str, str]]] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate_reply(
        self,
        user_message: str,
        conversation_id: str | None = None,
    ) -> LLMReply:
        effective_conversation_id = conversation_id or str(uuid.uuid4())
        transcript = self._sessions.get(effective_conversation_id, [])
        messages = [*transcript, {"role": "user", "content": user_message}]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Persist turn for follow-up calls on the same conversation_id.
        self._sessions[effective_conversation_id] = [
            *transcript,
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": content or ""},
        ]

        return LLMReply(
            message=content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            conversation_id=effective_conversation_id,
        )

    def clear_session(self, conversation_id: str) -> bool:
        if conversation_id in self._sessions:
            del self._sessions[conversation_id]
            return True
        return False
