import uuid

from fastapi import FastAPI

from app.schemas import ChatRequest, ChatResponse, Usage

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())
    return ChatResponse(
        conversation_id=conversation_id,
        message=f"You said: {request.message}",
        sources=[],
        usage=Usage(input_tokens=0, output_tokens=0),
    )
