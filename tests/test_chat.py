import re

from fastapi.testclient import TestClient

from app.llm import LLMReply
from app.main import app

client = TestClient(app)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def test_chat_new_conversation():
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert UUID_PATTERN.match(data["conversation_id"])
    assert data["message"] == "You said: hello"
    assert data["sources"] == []
    assert data["usage"]["input_tokens"] == 0
    assert data["usage"]["output_tokens"] == 0


def test_chat_existing_conversation():
    response = client.post(
        "/chat",
        json={"message": "hi", "conversation_id": "existing-id-123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "existing-id-123"
    assert data["message"] == "You said: hi"
    assert data["sources"] == []
    assert data["usage"]["input_tokens"] == 0
    assert data["usage"]["output_tokens"] == 0


def test_chat_empty_message_rejected():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_missing_message_rejected():
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_uses_stub_llm_by_default_even_when_openai_available(monkeypatch):
    class FakeLLM:
        def generate_reply(self, user_message: str) -> LLMReply:
            return LLMReply(
                message="openai would say this",
                input_tokens=1,
                output_tokens=1,
            )

    # Pretend OpenAI is available, but without setting real env vars.
    monkeypatch.setattr("app.main.make_llm", lambda: FakeLLM())

    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    data = response.json()
    # Should still use the stub echo behavior when no metadata.llm override is set.
    assert data["message"] == "You said: hello"


def test_chat_can_opt_in_to_openai_llm_when_requested(monkeypatch):
    class FakeLLM:
        def generate_reply(self, user_message: str) -> LLMReply:
            return LLMReply(
                message="openai-style reply",
                input_tokens=2,
                output_tokens=3,
            )

    monkeypatch.setattr("app.main.make_llm", lambda: FakeLLM())

    response = client.post(
        "/chat",
        json={"message": "hello", "metadata": {"llm": "openai"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "openai-style reply"