import uuid

from app.llm import LLMClient
from app.schemas import ChatRequest, ChatResponse, Usage


class ChatService:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def chat(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        reply = self._llm.generate_reply(request.message)
        return ChatResponse(
            conversation_id=conversation_id,
            message=reply.message,
            sources=[],
            usage=Usage(
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
            ),
        )
