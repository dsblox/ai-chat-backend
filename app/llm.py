from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMReply:
    message: str
    input_tokens: int
    output_tokens: int
    conversation_id: str | None = None


@dataclass
class ContextPolicy:
    max_cost_dollars: float
    summarization_prompt: str
    cost_per_input_token: float
    cost_per_output_token: float
    keep_recent_messages: int = 6


class LLMClient(Protocol):
    def generate_reply(
        self,
        user_message: str,
        conversation_id: str | None = None,
    ) -> LLMReply: ...

    def clear_session(self, conversation_id: str) -> bool: ...


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

    def clear_session(self, conversation_id: str) -> bool:
        return False
