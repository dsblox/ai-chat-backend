"""OpenAI chat completions client implementing the LLMClient protocol."""

import os
import uuid
from dataclasses import dataclass, field

from openai import OpenAI

from app.llm import ContextPolicy, LLMReply


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    summary: str | None = None
    summary_up_to: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class OpenAILLMClient:
    """Calls OpenAI chat completions API. Requires OPENAI_API_KEY in env."""

    # Shared in-memory transcript storage across client instances in-process.
    _sessions: dict[str, SessionState] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        context_policy: ContextPolicy | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._policy = context_policy

    def generate_reply(
        self,
        user_message: str,
        conversation_id: str | None = None,
    ) -> LLMReply:
        effective_conversation_id = conversation_id or str(uuid.uuid4())
        state = self._sessions.get(
            effective_conversation_id, SessionState()
        )

        context = self._build_context(state, user_message)

        sum_in = 0
        sum_out = 0
        if self._policy is not None:
            cost = self._estimate_input_cost(context)
            unsummarized_count = len(state.messages) - state.summary_up_to
            if (
                cost > self._policy.max_cost_dollars
                and unsummarized_count > self._policy.keep_recent_messages
            ):
                state, sum_in, sum_out = self._summarize(state)
                state.total_input_tokens += sum_in
                state.total_output_tokens += sum_out
                context = self._build_context(state, user_message)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=context,
        )
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        state.total_input_tokens += input_tokens
        state.total_output_tokens += output_tokens

        if self._policy:
            msg_cost = (
                (input_tokens + sum_in) * self._policy.cost_per_input_token
                + (output_tokens + sum_out) * self._policy.cost_per_output_token
            )
            session_cost = (
                state.total_input_tokens * self._policy.cost_per_input_token
                + state.total_output_tokens * self._policy.cost_per_output_token
            )
        else:
            msg_cost = 0.0
            session_cost = 0.0

        # Persist turn for follow-up calls on the same conversation_id.
        state.messages.append({"role": "user", "content": user_message})
        state.messages.append({"role": "assistant", "content": content or ""})
        self._sessions[effective_conversation_id] = state

        return LLMReply(
            message=content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            conversation_id=effective_conversation_id,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            cost=msg_cost,
            total_cost=session_cost,
        )

    def _build_context(
        self, state: SessionState, user_message: str
    ) -> list[dict[str, str]]:
        context: list[dict[str, str]] = []
        if state.summary is not None:
            context.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{state.summary}",
                }
            )
        context.extend(state.messages[state.summary_up_to:])
        context.append({"role": "user", "content": user_message})
        return context

    def _estimate_input_cost(self, messages: list[dict[str, str]]) -> float:
        assert self._policy is not None
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars / 4
        return estimated_tokens * self._policy.cost_per_input_token

    def _summarize(self, state: SessionState) -> tuple[SessionState, int, int]:
        assert self._policy is not None
        keep = self._policy.keep_recent_messages
        messages_to_summarize = state.messages[state.summary_up_to:-keep] if keep > 0 else state.messages[state.summary_up_to:]

        summarization_messages: list[dict[str, str]] = [
            {"role": "system", "content": self._policy.summarization_prompt},
        ]
        summarization_messages.extend(messages_to_summarize)
        if state.summary is not None:
            summarization_messages.append(
                {"role": "system", "content": f"Previous summary: {state.summary}"}
            )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=summarization_messages,
        )
        choice = response.choices[0] if response.choices else None
        new_summary = (
            choice.message.content if choice and choice.message else ""
        ) or ""
        usage = response.usage
        sum_input_tokens = usage.prompt_tokens if usage else 0
        sum_output_tokens = usage.completion_tokens if usage else 0

        new_summary_up_to = len(state.messages) - keep if keep > 0 else len(state.messages)
        new_state = SessionState(
            messages=state.messages,
            summary=new_summary,
            summary_up_to=new_summary_up_to,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
        )
        return new_state, sum_input_tokens, sum_output_tokens

    def clear_session(self, conversation_id: str) -> bool:
        if conversation_id in self._sessions:
            del self._sessions[conversation_id]
            return True
        return False
