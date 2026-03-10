"""Tests for OpenAILLMClient context window management."""

from unittest.mock import MagicMock, patch

from app.llm import ContextPolicy
from app.llm_openai import OpenAILLMClient, SessionState


def _make_policy(
    *,
    max_cost_dollars: float = 0.05,
    keep_recent_messages: int = 6,
) -> ContextPolicy:
    return ContextPolicy(
        max_cost_dollars=max_cost_dollars,
        summarization_prompt="Summarize this conversation.",
        cost_per_input_token=0.00000015,
        cost_per_output_token=0.0000006,
        keep_recent_messages=keep_recent_messages,
    )


def _mock_completion(content: str = "reply", input_tokens: int = 10, output_tokens: int = 5):
    """Return a mock matching the OpenAI ChatCompletion response shape."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_client(
    policy: ContextPolicy | None = None,
) -> OpenAILLMClient:
    """Create an OpenAILLMClient with a mocked OpenAI inner client."""
    # Reset shared class-level session state so tests are isolated.
    OpenAILLMClient._sessions = {}

    with patch("app.llm_openai.OpenAI"):
        client = OpenAILLMClient(
            api_key="fake-key",
            model="gpt-4o-mini",
            context_policy=policy,
        )
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = _mock_completion()
    return client


# ---------- Tests ----------


def test_no_summarization_below_threshold():
    """With a generous cost threshold, no summarization call should happen."""
    policy = _make_policy(max_cost_dollars=999.0)
    client = _make_client(policy)

    client.generate_reply("hello", conversation_id="c1")
    client.generate_reply("world", conversation_id="c1")
    client.generate_reply("foo", conversation_id="c1")

    # One API call per generate_reply, no extra summarization calls.
    assert client._client.chat.completions.create.call_count == 3


def test_summarization_triggered_above_threshold():
    """With a tiny cost threshold, summarization should trigger."""
    policy = _make_policy(max_cost_dollars=0.0000001, keep_recent_messages=2)
    client = _make_client(policy)

    # First call — context is tiny, no summarization yet.
    client._client.chat.completions.create.return_value = _mock_completion("r1")
    client.generate_reply("msg1", conversation_id="c1")
    assert client._client.chat.completions.create.call_count == 1

    # Second call — still only 2 unsummarized msgs, not > keep_recent_messages.
    client._client.chat.completions.create.return_value = _mock_completion("r2")
    client.generate_reply("msg2", conversation_id="c1")
    assert client._client.chat.completions.create.call_count == 2

    # Third call — now 4 unsummarized messages (msg1, r1, msg2, r2) > keep=2.
    # Summarization mock returns a summary, then the normal call happens.
    client._client.chat.completions.create.side_effect = [
        _mock_completion("the summary"),   # summarization call
        _mock_completion("r3"),            # normal reply call
    ]
    client.generate_reply("msg3", conversation_id="c1")
    # 2 previous + 2 new (summarize + reply) = 4 total
    assert client._client.chat.completions.create.call_count == 4

    # Verify the summarization call included the summarization prompt.
    summarize_call_args = client._client.chat.completions.create.call_args_list[2]
    summarize_messages = summarize_call_args.kwargs.get(
        "messages", summarize_call_args[1].get("messages", [])
    ) if summarize_call_args.kwargs else summarize_call_args[1]["messages"]
    assert any(
        m["role"] == "system" and "Summarize" in m["content"]
        for m in summarize_messages
    )

    # Fourth call — should include summary system message in context.
    client._client.chat.completions.create.side_effect = None
    client._client.chat.completions.create.return_value = _mock_completion("r4")
    client.generate_reply("msg4", conversation_id="c1")

    last_call_args = client._client.chat.completions.create.call_args
    last_messages = last_call_args.kwargs.get(
        "messages", last_call_args[1].get("messages", [])
    ) if last_call_args.kwargs else last_call_args[1]["messages"]
    assert any(
        m["role"] == "system" and m["content"].startswith("Summary of earlier conversation:")
        for m in last_messages
    )


def test_summary_replaces_old_messages_in_context():
    """After summarization, context should contain summary + recent msgs, not full transcript."""
    policy = _make_policy(max_cost_dollars=0.000001, keep_recent_messages=2)
    client = _make_client(policy)

    # Build up enough messages so summarization triggers on the third call.
    client._client.chat.completions.create.return_value = _mock_completion("r1")
    client.generate_reply("msg1", conversation_id="c1")

    client._client.chat.completions.create.return_value = _mock_completion("r2")
    client.generate_reply("msg2", conversation_id="c1")

    # Third call triggers summarization.
    client._client.chat.completions.create.side_effect = [
        _mock_completion("the summary"),
        _mock_completion("r3"),
    ]
    client.generate_reply("msg3", conversation_id="c1")

    # Fourth call — inspect the context sent to the API.
    client._client.chat.completions.create.side_effect = None
    client._client.chat.completions.create.return_value = _mock_completion("r4")
    client.generate_reply("msg4", conversation_id="c1")

    last_call_args = client._client.chat.completions.create.call_args
    last_messages = last_call_args.kwargs.get(
        "messages", last_call_args[1].get("messages", [])
    ) if last_call_args.kwargs else last_call_args[1]["messages"]

    # Should be: [summary_system_msg, recent_messages..., new_user_msg]
    # NOT the full transcript of msg1, r1, msg2, r2, msg3, r3, msg4.
    assert last_messages[0]["role"] == "system"
    assert last_messages[0]["content"].startswith("Summary of earlier conversation:")

    # The full transcript has 6 messages (3 turns * 2) before msg4.
    # With keep_recent_messages=2 and summarization, context should be much smaller.
    # Context = summary_msg + 2 recent msgs + new user msg = 4 at most
    # (could be more if a second round of keep_recent is different, but definitely < 8)
    assert len(last_messages) < 8, (
        f"Expected trimmed context but got {len(last_messages)} messages"
    )


def test_clear_session_works_with_session_state():
    """clear_session should work with the SessionState-based storage."""
    client = _make_client(policy=_make_policy())

    client._client.chat.completions.create.return_value = _mock_completion()
    client.generate_reply("hello", conversation_id="c1")

    assert client.clear_session("c1") is True
    assert "c1" not in client._sessions


def test_no_summarization_without_policy():
    """With context_policy=None, no summarization should ever happen."""
    client = _make_client(policy=None)

    # Send many messages — none should trigger summarization.
    for i in range(10):
        client._client.chat.completions.create.return_value = _mock_completion(f"r{i}")
        client.generate_reply(f"message {i}", conversation_id="c1")

    # Exactly 10 calls — one per generate_reply, no summarization.
    assert client._client.chat.completions.create.call_count == 10
