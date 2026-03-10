import uuid

from app.llm import LLMClient
from app.schemas import ChatRequest, ChatResponse, Usage


class ChatService:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def chat(self, request: ChatRequest) -> ChatResponse:
        # Keep compatibility with simple one-arg test doubles used in tests.
        try:
            reply = self._llm.generate_reply(
                request.message,
                conversation_id=request.conversation_id,
            )
        except TypeError:
            reply = self._llm.generate_reply(request.message)

        conversation_id = (
            request.conversation_id
            or reply.conversation_id
            or str(uuid.uuid4())
        )
        return ChatResponse(
            conversation_id=conversation_id,
            message=reply.message,
            sources=[],
            usage=Usage(
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
            ),
            total_usage=Usage(
                input_tokens=reply.total_input_tokens,
                output_tokens=reply.total_output_tokens,
            ),
            cost=reply.cost,
            total_cost=reply.total_cost,
        )
