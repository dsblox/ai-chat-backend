from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    sources: list = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    total_usage: Usage = Field(default_factory=Usage)
    cost: float = 0.0
    total_cost: float = 0.0


class ResetResponse(BaseModel):
    conversation_id: str
    cleared: bool
