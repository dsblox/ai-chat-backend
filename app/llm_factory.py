"""Factory for selecting the LLM client based on environment."""

import os

from app.llm import StubLLMClient
from app.llm_openai import OpenAILLMClient


def make_llm(env: dict[str, str] | None = None):
    """Return OpenAILLMClient if OPENAI_API_KEY is set, else StubLLMClient."""
    e = env if env is not None else os.environ
    if e.get("OPENAI_API_KEY"):
        return OpenAILLMClient()
    return StubLLMClient()
