from fastapi import FastAPI

from app.chat_service import ChatService
from app.llm import StubLLMClient
from app.schemas import ChatRequest, ChatResponse

app = FastAPI()
chat_service = ChatService(llm=StubLLMClient())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return chat_service.chat(request)
