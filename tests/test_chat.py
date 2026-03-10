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
    assert data["total_usage"]["input_tokens"] == 0
    assert data["total_usage"]["output_tokens"] == 0


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
    assert data["total_usage"]["input_tokens"] == 0
    assert data["total_usage"]["output_tokens"] == 0


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


def test_chat_openai_conversation_id_is_returned_and_reused(monkeypatch):
    class FakeSessionLLM:
        def __init__(self) -> None:
            self._calls_by_conversation: dict[str, int] = {}
            self._totals: dict[str, tuple[int, int]] = {}
            self._counter = 0

        def generate_reply(
            self,
            user_message: str,
            conversation_id: str | None = None,
        ) -> LLMReply:
            if conversation_id is None:
                self._counter += 1
                conversation_id = f"openai-session-{self._counter}"
            self._calls_by_conversation[conversation_id] = (
                self._calls_by_conversation.get(conversation_id, 0) + 1
            )
            prev_in, prev_out = self._totals.get(conversation_id, (0, 0))
            total_in = prev_in + 1
            total_out = prev_out + 1
            self._totals[conversation_id] = (total_in, total_out)
            call_number = self._calls_by_conversation[conversation_id]
            return LLMReply(
                message=f"{conversation_id}#{call_number}:{user_message}",
                input_tokens=1,
                output_tokens=1,
                conversation_id=conversation_id,
                total_input_tokens=total_in,
                total_output_tokens=total_out,
            )

        def clear_session(self, conversation_id: str) -> bool:
            if conversation_id in self._calls_by_conversation:
                del self._calls_by_conversation[conversation_id]
                self._totals.pop(conversation_id, None)
                return True
            return False

    fake = FakeSessionLLM()
    monkeypatch.setattr("app.main.make_llm", lambda: fake)

    first = client.post(
        "/chat",
        json={"message": "hello", "metadata": {"llm": "openai"}},
    )
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["conversation_id"] == "openai-session-1"
    assert first_data["message"] == "openai-session-1#1:hello"
    assert first_data["total_usage"]["input_tokens"] == 1
    assert first_data["total_usage"]["output_tokens"] == 1

    second = client.post(
        "/chat",
        json={
            "message": "again",
            "conversation_id": first_data["conversation_id"],
            "metadata": {"llm": "openai"},
        },
    )
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["conversation_id"] == "openai-session-1"
    assert second_data["message"] == "openai-session-1#2:again"
    assert second_data["total_usage"]["input_tokens"] == 2
    assert second_data["total_usage"]["output_tokens"] == 2


def test_reset_conversation_stub_returns_not_cleared():
    response = client.delete("/conversations/some-id")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "some-id"
    assert data["cleared"] is False


def test_reset_conversation_openai_clears_session(monkeypatch):
    class FakeSessionLLM:
        def __init__(self) -> None:
            self._calls_by_conversation: dict[str, int] = {}
            self._totals: dict[str, tuple[int, int]] = {}
            self._counter = 0

        def generate_reply(
            self,
            user_message: str,
            conversation_id: str | None = None,
        ) -> LLMReply:
            if conversation_id is None:
                self._counter += 1
                conversation_id = f"openai-session-{self._counter}"
            self._calls_by_conversation[conversation_id] = (
                self._calls_by_conversation.get(conversation_id, 0) + 1
            )
            prev_in, prev_out = self._totals.get(conversation_id, (0, 0))
            total_in = prev_in + 1
            total_out = prev_out + 1
            self._totals[conversation_id] = (total_in, total_out)
            call_number = self._calls_by_conversation[conversation_id]
            return LLMReply(
                message=f"{conversation_id}#{call_number}:{user_message}",
                input_tokens=1,
                output_tokens=1,
                conversation_id=conversation_id,
                total_input_tokens=total_in,
                total_output_tokens=total_out,
            )

        def clear_session(self, conversation_id: str) -> bool:
            if conversation_id in self._calls_by_conversation:
                del self._calls_by_conversation[conversation_id]
                self._totals.pop(conversation_id, None)
                return True
            return False

    fake = FakeSessionLLM()
    monkeypatch.setattr("app.main.make_llm", lambda: fake)

    # Build up a session with two messages.
    first = client.post(
        "/chat",
        json={"message": "hello", "metadata": {"llm": "openai"}},
    )
    conv_id = first.json()["conversation_id"]

    client.post(
        "/chat",
        json={
            "message": "again",
            "conversation_id": conv_id,
            "metadata": {"llm": "openai"},
        },
    )

    # Clear the session.
    reset = client.delete(f"/conversations/{conv_id}?llm=openai")
    assert reset.status_code == 200
    reset_data = reset.json()
    assert reset_data["conversation_id"] == conv_id
    assert reset_data["cleared"] is True

    # Next message on same conversation_id should show call #1 (counter reset).
    third = client.post(
        "/chat",
        json={
            "message": "fresh",
            "conversation_id": conv_id,
            "metadata": {"llm": "openai"},
        },
    )
    assert third.json()["message"] == f"{conv_id}#1:fresh"


def test_reset_conversation_nonexistent_returns_not_cleared(monkeypatch):
    class FakeSessionLLM:
        def __init__(self) -> None:
            self._calls_by_conversation: dict[str, int] = {}
            self._totals: dict[str, tuple[int, int]] = {}
            self._counter = 0

        def generate_reply(
            self,
            user_message: str,
            conversation_id: str | None = None,
        ) -> LLMReply:
            if conversation_id is None:
                self._counter += 1
                conversation_id = f"openai-session-{self._counter}"
            self._calls_by_conversation[conversation_id] = (
                self._calls_by_conversation.get(conversation_id, 0) + 1
            )
            prev_in, prev_out = self._totals.get(conversation_id, (0, 0))
            total_in = prev_in + 1
            total_out = prev_out + 1
            self._totals[conversation_id] = (total_in, total_out)
            call_number = self._calls_by_conversation[conversation_id]
            return LLMReply(
                message=f"{conversation_id}#{call_number}:{user_message}",
                input_tokens=1,
                output_tokens=1,
                conversation_id=conversation_id,
                total_input_tokens=total_in,
                total_output_tokens=total_out,
            )

        def clear_session(self, conversation_id: str) -> bool:
            if conversation_id in self._calls_by_conversation:
                del self._calls_by_conversation[conversation_id]
                self._totals.pop(conversation_id, None)
                return True
            return False

    fake = FakeSessionLLM()
    monkeypatch.setattr("app.main.make_llm", lambda: fake)

    response = client.delete("/conversations/does-not-exist?llm=openai")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "does-not-exist"
    assert data["cleared"] is False
