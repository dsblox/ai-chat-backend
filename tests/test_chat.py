import re

from fastapi.testclient import TestClient

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