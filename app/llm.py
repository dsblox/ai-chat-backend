from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMReply:
    message: str
    input_tokens: int
    output_tokens: int
    conversation_id: str | None = None


class LLMClient(Protocol):
    def generate_reply(
        self,
        user_message: str,
        conversation_id: str | None = None,
    ) -> LLMReply: ...


class StubLLMClient:
    def generate_reply(
        self,
        user_message: str,
        conversation_id: str | None = None,
    ) -> LLMReply:
        return LLMReply(
            message=f"You said: {user_message}",
            input_tokens=0,
            output_tokens=0,
            conversation_id=conversation_id,
        )
