"""Tests for LLM client selection based on environment."""

from app.llm import StubLLMClient
from app.llm_factory import make_llm
from app.llm_openai import OpenAILLMClient


def test_make_llm_uses_stub_when_api_key_unset():
    llm = make_llm(env={})
    assert isinstance(llm, StubLLMClient)


def test_make_llm_uses_openai_client_when_api_key_set():
    llm = make_llm(env={"OPENAI_API_KEY": "sk-test-key"})
    assert isinstance(llm, OpenAILLMClient)
    assert not isinstance(llm, StubLLMClient)
