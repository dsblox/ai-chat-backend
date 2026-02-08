from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMReply:
    message: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    def generate_reply(self, user_message: str) -> LLMReply: ...


class StubLLMClient:
    def generate_reply(self, user_message: str) -> LLMReply:
        return LLMReply(
            message=f"You said: {user_message}",
            input_tokens=0,
            output_tokens=0,
        )
