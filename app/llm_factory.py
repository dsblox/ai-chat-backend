"""Factory for selecting the LLM client based on environment."""

import os

from app.llm import ContextPolicy, StubLLMClient
from app.llm_openai import OpenAILLMClient

DEFAULT_SUMMARIZATION_PROMPT = (
    "You are a conversation summarizer. Condense the following conversation "
    "into a concise summary that preserves: key facts, names, decisions, "
    "technical details, and the overall topic/goal. Do not add information "
    "that was not discussed. Write in third person past tense."
)

_TOKEN_COSTS: dict[str, tuple[float, float]] = {
    # (input $/token, output $/token)
    "gpt-4o-mini": (0.00000015, 0.0000006),
    "gpt-4o": (0.0000025, 0.00001),
}

_DEFAULT_COSTS = _TOKEN_COSTS["gpt-4o-mini"]


def _input_cost(model: str) -> float:
    return _TOKEN_COSTS.get(model, _DEFAULT_COSTS)[0]


def _output_cost(model: str) -> float:
    return _TOKEN_COSTS.get(model, _DEFAULT_COSTS)[1]


def make_llm(env: dict[str, str] | None = None):
    """Return OpenAILLMClient if OPENAI_API_KEY is set, else StubLLMClient."""
    e = env if env is not None else os.environ
    if e.get("OPENAI_API_KEY"):
        model = e.get("OPENAI_MODEL", "gpt-4o-mini")
        max_cost = float(e.get("CONTEXT_MAX_COST", "0.05"))
        policy = ContextPolicy(
            max_cost_dollars=max_cost,
            summarization_prompt=DEFAULT_SUMMARIZATION_PROMPT,
            cost_per_input_token=_input_cost(model),
            cost_per_output_token=_output_cost(model),
        )
        return OpenAILLMClient(model=model, context_policy=policy)
    return StubLLMClient()
